import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import os
import numpy as np
import math

class HandTrackingMouseControlSimpleDisplay:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.5, 
            max_num_hands=1, 
            static_image_mode=False        )
        
        time.sleep(0.001)
        os.system("cls" if os.name == "nt" else "clear")
        print("press strg or ctrl + c to exit")
        print("Zeige mit dem Zeigefinger nach links/rechts um zwischen Monitoren zu wechseln")

        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is available
        if not self.cap.isOpened():
            print("Error: Kamera konnte nicht geöffnet werden!")
            raise Exception("Camera not available")        # Multi-Monitor Setup
        self.setup_multi_monitor()
        print(f"Erkannte Monitore: {len(self.monitors)}")
        for i, monitor in enumerate(self.monitors):
            print(f"  Monitor {i+1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
            if hasattr(monitor, 'is_primary'):
                print(f"    Primary: {monitor.is_primary}")
        print(f"Gesamtbildschirmgröße: {self.total_screen_width}x{self.total_screen_height}")
        
        self.camera_width, self.camera_height = 640, 480
        
        # Define regions for different monitors/functions
        self.region_x_min, self.region_x_max = 0.2, 0.8
        self.region_y_min, self.region_y_max = 0.2, 0.8
        
        # Monitor switching parameters
        self.current_monitor = 0
          # Directional monitor switching
        self.directional_switch_active = False
        self.pointing_threshold = 0.07
        self.pointing_depth_threshold = 0.06
        self.last_pointing_switch = 0
        self.pointing_switch_cooldown = 0.5  # Faster switching
        
        # Monitor zones for directional switching
        self.monitor_zones = self.setup_monitor_zones()
        
        # Visual feedback
        self.monitor_highlight_duration = 2.0
        self.monitor_highlight_start = 0
        self.highlighted_monitor = -1
          # 3D tracking parameters (more sensitive)
        self.z_baseline = None
        self.z_smoothing_factor = 0.8  # Higher for more stability
        self.smoothed_z = 0
        self.depth_scale = 100  # Scale factor for depth visualization          # Click detection (pure 3D)
        self.clicking = False
        self.click_start_time = None
        self.holding = False
        self.last_gesture_time = 0
        self.gesture_delay = 0.05  # Faster response
        
        # Position smoothing
        self.position_history = []
        self.position_smoothing = 3  # Less smoothing for more responsive
        
        # 3D distance for clicking (optimized for 3D only)
        self.pinch_threshold_3d = 0.04  # Very tight threshold for precise 3D control
        self.pinch_release_threshold_3d = 0.06
        
        pyautogui.FAILSAFE = False
        
        # Start threads
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
        
        self.monitor_border_thread = threading.Thread(target=self.draw_monitor_borders, daemon=True)
        self.monitor_border_thread.start()
        
        # Initialize control region based on current monitor
        self.update_control_region_for_monitor()
        
    def setup_multi_monitor(self):
        """Setup multi-monitor configuration with comprehensive debugging."""
        try:
            from screeninfo import get_monitors
            raw_monitors = list(get_monitors())
            
            print("=== RAW MONITOR DETECTION ===")
            for i, monitor in enumerate(raw_monitors):
                print(f"  Raw Monitor {i}: {monitor}")
                print(f"    Size: {monitor.width}x{monitor.height}")
                print(f"    Position: ({monitor.x}, {monitor.y})")
                if hasattr(monitor, 'is_primary'):
                    print(f"    Primary: {monitor.is_primary}")
                if hasattr(monitor, 'name'):
                    print(f"    Name: {monitor.name}")
            
            # Sort monitors by X position (left to right) for consistent ordering
            self.monitors = sorted(raw_monitors, key=lambda m: m.x)
            
            print("\n=== SORTED MONITORS (left to right) ===")
            for i, monitor in enumerate(self.monitors):
                print(f"  Monitor {i}: Size {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
                if hasattr(monitor, 'is_primary'):
                    print(f"    Primary: {monitor.is_primary}")
                if hasattr(monitor, 'name'):
                    print(f"    Name: {monitor.name}")
            
            # Calculate virtual desktop bounds properly
            min_x = min(m.x for m in self.monitors)
            min_y = min(m.y for m in self.monitors)
            max_x = max(m.x + m.width for m in self.monitors)
            max_y = max(m.y + m.height for m in self.monitors)
            
            self.total_screen_width = max_x - min_x
            self.total_screen_height = max_y - min_y
            self.virtual_desktop_offset_x = min_x
            self.virtual_desktop_offset_y = min_y
            
            print(f"\n=== VIRTUAL DESKTOP CALCULATION ===")
            print(f"  Min bounds: ({min_x}, {min_y})")
            print(f"  Max bounds: ({max_x}, {max_y})")
            print(f"  Total size: {self.total_screen_width}x{self.total_screen_height}")
            print(f"  Virtual offset: ({self.virtual_desktop_offset_x}, {self.virtual_desktop_offset_y})")
            
            # Find primary monitor index
            self.primary_monitor_index = 0
            for i, monitor in enumerate(self.monitors):
                if hasattr(monitor, 'is_primary') and monitor.is_primary:
                    self.primary_monitor_index = i
                    break
            
            self.primary_monitor = self.monitors[self.primary_monitor_index]
            print(f"\n=== PRIMARY MONITOR ===")
            print(f"  Primary monitor index: {self.primary_monitor_index}")
            print(f"  Primary monitor: {self.primary_monitor.width}x{self.primary_monitor.height} at ({self.primary_monitor.x}, {self.primary_monitor.y})")
            
        except ImportError:
            print("WARNING: screeninfo not available, using fallback")
            self.screen_width, self.screen_height = pyautogui.size()
            self.monitors = [type('Monitor', (), {
                'x': 0, 'y': 0, 
                'width': self.screen_width, 
                'height': self.screen_height,
                'is_primary': True
            })]
            self.total_screen_width = self.screen_width
            self.total_screen_height = self.screen_height
            self.virtual_desktop_offset_x = 0
            self.virtual_desktop_offset_y = 0
            self.primary_monitor = self.monitors[0]
            self.primary_monitor_index = 0
    
    def setup_monitor_zones(self):
        """Setup zones for directional monitor switching."""
        zones = []
        if len(self.monitors) <= 1:
            return zones
        
        # Sort monitors by X position (left to right)
        sorted_monitors = sorted(enumerate(self.monitors), key=lambda x: x[1].x)
        
        # Create directional zones
        for i, (original_index, monitor) in enumerate(sorted_monitors):
            total_width = sum(m.width for _, m in sorted_monitors)
            zone_start = sum(m.width for _, m in sorted_monitors[:i]) / total_width
            zone_end = sum(m.width for _, m in sorted_monitors[:i+1]) / total_width
            
            zone = {
                'monitor_index': original_index,
                'x_min': zone_start,
                'x_max': zone_end,
                'monitor': monitor,
                'position': i
            }
            zones.append(zone)
        
        return zones
    
    def detect_pointing_gesture(self, hand_landmarks):
        """Detect pointing gesture for directional monitor switching."""
        current_time = time.time()
        
        if current_time - self.last_pointing_switch < self.pointing_switch_cooldown:
            return
        
        if len(self.monitors) <= 1:
            return
        
        # Get hand position (use same finger as mouse control)
        middle_finger_control = hand_landmarks.landmark[9]
        
        # Use the SAME normalization as mouse control
        norm_x = (middle_finger_control.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
        norm_x = max(0, min(1, norm_x))  # Clamp to 0-1
        
        # Debug output
        # print(f"Monitor switching: norm_x={norm_x:.2f}, raw_x={middle_finger_control.x:.2f}")        # Very simple detection - check normalized position within control region
        if norm_x < 0.1:  # Far left within control region
            # Only switch left if there's a monitor to the left
            current_monitor_obj = self.monitors[self.current_monitor]
            left_monitor_index = None
            
            # Find the monitor directly to the left
            for i, monitor in enumerate(self.monitors):
                if (monitor.x + monitor.width <= current_monitor_obj.x and 
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if left_monitor_index is None or monitor.x > self.monitors[left_monitor_index].x:
                        left_monitor_index = i
            
            if left_monitor_index is not None:
                self.current_monitor = left_monitor_index
                self.last_pointing_switch = current_time
                print(f">>> SWITCHED TO LEFT MONITOR {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
                
        elif norm_x > 0.9:  # Far right within control region
            # Only switch right if there's a monitor to the right
            current_monitor_obj = self.monitors[self.current_monitor]
            right_monitor_index = None
            
            # Find the monitor directly to the right
            for i, monitor in enumerate(self.monitors):
                if (monitor.x >= current_monitor_obj.x + current_monitor_obj.width and
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if right_monitor_index is None or monitor.x < self.monitors[right_monitor_index].x:
                        right_monitor_index = i
            
            if right_monitor_index is not None:
                self.current_monitor = right_monitor_index
                self.last_pointing_switch = current_time
                print(f">>> SWITCHED TO RIGHT MONITOR {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
    
    def get_monitor_from_pointing_direction(self, hand_x):
        """Determine target monitor based on pointing direction."""
        if len(self.monitors) <= 1:
            return 0
        
        for zone in self.monitor_zones:
            if zone['x_min'] <= hand_x <= zone['x_max']:
                return zone['monitor_index']
        
        if hand_x < 0.3:
            return min(range(len(self.monitors)), key=lambda i: self.monitors[i].x)
        elif hand_x > 0.7:
            return max(range(len(self.monitors)), key=lambda i: self.monitors[i].x)        
        return self.current_monitor
    
    def draw_monitor_borders(self):
        """Draw permanent borders around the current active monitor with enhanced debugging."""
        try:
            import tkinter as tk
            
            current_windows = []
            last_monitor = -1
            
            while True:
                # Check if monitor changed or we need to create borders
                if self.current_monitor != last_monitor:
                    # Clean up old windows first
                    for window in current_windows:
                        try:
                            window.destroy()
                        except:
                            pass
                    current_windows = []
                    
                    # Create new borders for current monitor
                    monitor = self.monitors[self.current_monitor]
                    print(f"\n>>> CREATING BORDER ON MONITOR {self.current_monitor + 1} <<<")
                    print(f"  Monitor size: {monitor.width}x{monitor.height}")
                    print(f"  Monitor position: ({monitor.x}, {monitor.y})")
                    print(f"  Border will cover area: ({monitor.x}, {monitor.y}) to ({monitor.x + monitor.width}, {monitor.y + monitor.height})")
                    
                    try:
                        # Create invisible root window
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-alpha', 0.0)
                        
                        border_width = 8  # Increased border width for better visibility
                        border_color = '#0080FF'  # Bright blue border
                        
                        # Calculate border positions
                        top_x, top_y = monitor.x, monitor.y
                        bottom_x, bottom_y = monitor.x, monitor.y + monitor.height - border_width
                        left_x, left_y = monitor.x, monitor.y
                        right_x, right_y = monitor.x + monitor.width - border_width, monitor.y
                        
                        print(f"  Border positions:")
                        print(f"    Top: {monitor.width}x{border_width} at ({top_x}, {top_y})")
                        print(f"    Bottom: {monitor.width}x{border_width} at ({bottom_x}, {bottom_y})")
                        print(f"    Left: {border_width}x{monitor.height} at ({left_x}, {left_y})")
                        print(f"    Right: {border_width}x{monitor.height} at ({right_x}, {right_y})")
                        
                        # Top border
                        top = tk.Toplevel(root)
                        top.geometry(f"{monitor.width}x{border_width}+{top_x}+{top_y}")
                        top.configure(bg=border_color)
                        top.overrideredirect(True)
                        top.attributes('-topmost', True)
                        top.attributes('-alpha', 0.8)
                        current_windows.append(top)
                        
                        # Bottom border
                        bottom = tk.Toplevel(root)
                        bottom.geometry(f"{monitor.width}x{border_width}+{bottom_x}+{bottom_y}")
                        bottom.configure(bg=border_color)
                        bottom.overrideredirect(True)
                        bottom.attributes('-topmost', True)
                        bottom.attributes('-alpha', 0.8)
                        current_windows.append(bottom)
                        
                        # Left border
                        left = tk.Toplevel(root)
                        left.geometry(f"{border_width}x{monitor.height}+{left_x}+{left_y}")
                        left.configure(bg=border_color)
                        left.overrideredirect(True)
                        left.attributes('-topmost', True)
                        left.attributes('-alpha', 0.8)
                        current_windows.append(left)
                        
                        # Right border
                        right = tk.Toplevel(root)
                        right.geometry(f"{border_width}x{monitor.height}+{right_x}+{right_y}")
                        right.configure(bg=border_color)
                        right.overrideredirect(True)
                        right.attributes('-topmost', True)
                        right.attributes('-alpha', 0.8)
                        current_windows.append(right)
                        
                        current_windows.append(root)
                        last_monitor = self.current_monitor
                        print(f"  ✓ Border windows created successfully")
                        
                    except Exception as e:
                        print(f"  ✗ Border creation error: {e}")
                
                # Keep borders alive and prevent flickering
                try:
                    if current_windows and current_windows[-1]:  # root window
                        current_windows[-1].update()
                except:
                    pass
                
                time.sleep(0.1)
        except Exception as e:
            print(f"Border thread error: {e}")
            time.sleep(1)
    
    def convert_to_monitor_coordinates(self, norm_x, norm_y, monitor_index):
        """Convert normalized coordinates to specific monitor coordinates with comprehensive debugging."""
        if monitor_index >= len(self.monitors):
            print(f"WARNING: Invalid monitor index {monitor_index}, using 0")
            monitor_index = 0
        
        monitor = self.monitors[monitor_index]
        
        # Calculate absolute coordinates within the specific monitor
        local_x = norm_x * monitor.width
        local_y = norm_y * monitor.height
        
        # Add monitor offset to get global coordinates
        global_x = int(local_x + monitor.x)
        global_y = int(local_y + monitor.y)
        
        # Debug output for coordinates
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        # Print debug info every 60 frames (2 seconds at 30fps) to avoid spam
        if self.debug_counter % 60 == 0:
            print(f"\n=== COORDINATE CONVERSION DEBUG ===")
            print(f"  Target Monitor {monitor_index}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
            print(f"  Normalized input: ({norm_x:.3f}, {norm_y:.3f})")
            print(f"  Local coords: ({local_x:.1f}, {local_y:.1f})")
            print(f"  Global coords: ({global_x}, {global_y})")
            
            # Verify coordinates are within monitor bounds
            if monitor.x <= global_x <= monitor.x + monitor.width and monitor.y <= global_y <= monitor.y + monitor.height:
                print(f"  ✓ Coordinates are within monitor bounds")
            else:
                print(f"  ⚠ Coordinates are OUTSIDE monitor bounds!")
                print(f"    Monitor bounds: x=[{monitor.x}, {monitor.x + monitor.width}], y=[{monitor.y}, {monitor.y + monitor.height}]")
        
        return global_x, global_y
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two landmarks."""
        return math.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2
        )
    
    def smooth_position(self, new_position):
        """Smooth mouse position using moving average."""
        self.position_history.append(new_position)
        if len(self.position_history) > self.position_smoothing:
            self.position_history.pop(0)
        
        avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        return (int(avg_x), int(avg_y))
    
    def analyze_hand_depth(self, hand_landmarks):
        """Analyze hand depth."""
        palm_base = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        
        palm_z = (palm_base.z + middle_mcp.z) / 2
        
        if self.z_baseline is None:
            self.z_baseline = palm_z
            return 0
        
        relative_depth = palm_z - self.z_baseline
        self.smoothed_z = self.z_smoothing_factor * self.smoothed_z + (1 - self.z_smoothing_factor) * relative_depth
        
        return self.smoothed_z
    
    def detect_3d_click(self, hand_landmarks):
        """Pure 3D click detection using only 3D distance between thumb and index finger."""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_delay:
            return
        
        if not self.directional_switch_active:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
              # Calculate only 3D distance
            pinch_distance_3d = self.calculate_3d_distance(thumb_tip, index_tip)
            
            if pinch_distance_3d < self.pinch_threshold_3d:
                if not self.clicking:
                    self.clicking = True
                    self.holding = True
                    pyautogui.mouseDown()
                    self.last_gesture_time = current_time
                    print(f"3D Click! Distance: {pinch_distance_3d:.3f}")
            
            elif pinch_distance_3d > self.pinch_release_threshold_3d:
                if self.clicking and self.holding:
                    pyautogui.mouseUp()
                    self.clicking = False
                    self.holding = False
                    self.last_gesture_time = current_time
                    print(f"3D Release! Distance: {pinch_distance_3d:.3f}")

    def handle_mouse(self):
        """Function to handle mouse actions in a separate thread."""
        while True:
            if hasattr(self, 'mouse_position') and hasattr(self, 'mouse_action'):
                if self.mouse_action == "move":
                    pyautogui.moveTo(*self.mouse_position)
                self.mouse_action = None
            time.sleep(0.001)

    def update_control_region_for_monitor(self):
        """Update control region based on current monitor's actual pixel size and orientation."""
        if not self.monitors or self.current_monitor >= len(self.monitors):
            print(f"WARNING: Invalid monitor configuration. Current monitor: {self.current_monitor}, Available monitors: {len(self.monitors)}")
            if self.monitors:
                self.current_monitor = 0  # Reset to first monitor
                print(f"Reset to monitor 0")
            else:
                print("No monitors available, skipping control region update")
                return
        
        current_monitor = self.monitors[self.current_monitor]
        monitor_width = current_monitor.width
        monitor_height = current_monitor.height
        monitor_aspect_ratio = monitor_width / monitor_height
        
        print(f"\n=== CONTROL REGION UPDATE ===")
        print(f"  Current monitor: {monitor_width}x{monitor_height}")
        print(f"  Monitor aspect ratio: {monitor_aspect_ratio:.3f}")
        
        # Camera resolution (fixed)
        camera_aspect_ratio = self.camera_width / self.camera_height
        print(f"  Camera aspect ratio: {camera_aspect_ratio:.3f}")
          # Calculate scale factor based on actual pixel dimensions
        # Compare this monitor to a reference monitor (the largest one)
        largest_monitor_area = max(m.width * m.height for m in self.monitors)
        monitor_area = monitor_width * monitor_height
        area_ratio = monitor_area / largest_monitor_area
        
        print(f"  Monitor area: {monitor_area} pixels")
        print(f"  Largest monitor area: {largest_monitor_area} pixels")
        print(f"  Area ratio vs largest: {area_ratio:.3f}")        # Base control region size - scale more aggressively based on monitor size
        base_margin = 0.1  # 10% border around the control region - good balance
        
        # Scale the control region size based on monitor pixel dimensions
        # More generous scaling to make the region larger
        size_scale = 0.6 + (area_ratio * 0.3)  # Scale between 60% and 90%
        
        print(f"  Size scale factor: {size_scale:.3f}")
        
        # Calculate actual control region dimensions proportional to monitor
        # Always maintain the monitor's aspect ratio
        region_width = (1.0 - 2 * base_margin) * size_scale
        region_height = region_width / monitor_aspect_ratio
        
        # If the height would exceed camera bounds, scale down proportionally
        max_height = 1.0 - 2 * base_margin
        if region_height > max_height:
            region_height = max_height * size_scale
            region_width = region_height * monitor_aspect_ratio
        
        # If the width would exceed camera bounds, scale down proportionally
        max_width = 1.0 - 2 * base_margin
        if region_width > max_width:
            region_width = max_width * size_scale
            region_height = region_width / monitor_aspect_ratio        # Center the region
        horizontal_center = 0.5
        vertical_center = 0.5
        
        self.region_x_min = horizontal_center - region_width / 2
        self.region_x_max = horizontal_center + region_width / 2
        self.region_y_min = vertical_center - region_height / 2
        self.region_y_max = vertical_center + region_height / 2        # Ensure bounds are within camera frame with safe margins
        self.region_x_min = max(0.08, self.region_x_min)  # Minimum 8% margin
        self.region_x_max = min(0.92, self.region_x_max)  # Maximum 92% of camera width
        self.region_y_min = max(0.08, self.region_y_min)  # Minimum 8% margin
        self.region_y_max = min(0.92, self.region_y_max)  # Maximum 92% of camera height
        
        print(f"  Size scale factor: {size_scale:.3f}")
        print(f"  New control region: X({self.region_x_min:.3f}-{self.region_x_max:.3f}), Y({self.region_y_min:.3f}-{self.region_y_max:.3f})")
        print(f"  Control region size: {(self.region_x_max - self.region_x_min):.3f} x {(self.region_y_max - self.region_y_min):.3f}")
        control_aspect = (self.region_x_max - self.region_x_min) / (self.region_y_max - self.region_y_min)
        print(f"  Control region aspect ratio: {control_aspect:.3f}")
        print(f"  Expected monitor aspect ratio: {monitor_aspect_ratio:.3f}")
        print(f"  Aspect ratio match: {abs(control_aspect - monitor_aspect_ratio) < 0.1}")
    
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Use middle finger MCP (joint 9) instead of tip for more stable control
                middle_finger_control = hand_landmarks.landmark[9]  # Like original
                
                # Limit detection to defined region
                norm_x = (middle_finger_control.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
                norm_y = (middle_finger_control.y - self.region_y_min) / (self.region_y_max - self.region_y_min)
                
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                
                # Convert to current monitor coordinates
                x, y = self.convert_to_monitor_coordinates(norm_x, norm_y, self.current_monitor)
                
                # Smooth position
                raw_position = (x, y)
                self.mouse_position = self.smooth_position(raw_position)
                self.mouse_action = "move"
                
                # Gesture detection
                self.detect_pointing_gesture(hand_landmarks)
                self.detect_3d_click(hand_landmarks)
                
                # Draw hand landmarks - like the original
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
              # Draw control region and 3D depth info
            region_start_x = int(self.region_x_min * self.camera_width)
            region_end_x = int(self.region_x_max * self.camera_width)
            region_start_y = int(self.region_y_min * self.camera_height)
            region_end_y = int(self.region_y_max * self.camera_height)
            
            cv2.rectangle(frame, (region_start_x, region_start_y), 
                         (region_end_x, region_end_y), (0, 255, 0), 2)            # Show 3D depth and pinch distance information
            if hasattr(self, 'smoothed_z'):
                depth_text = f"Depth: {self.smoothed_z:.3f}"
                cv2.putText(frame, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show current monitor info
                monitor_text = f"Monitor: {self.current_monitor + 1}/{len(self.monitors)}"
                cv2.putText(frame, monitor_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Visual depth indicator
                depth_bar_length = int(abs(self.smoothed_z) * self.depth_scale)
                depth_bar_length = min(200, max(0, depth_bar_length))
                color = (0, 255, 0) if self.smoothed_z < 0 else (0, 0, 255)
                cv2.rectangle(frame, (10, 80), (10 + depth_bar_length, 100), color, -1)
                  # Show 3D pinch distance if hand is detected
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    pinch_distance = self.calculate_3d_distance(thumb_tip, index_tip)
                    
                    pinch_text = f"3D Pinch: {pinch_distance:.3f}"
                    cv2.putText(frame, pinch_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Visual pinch indicator
                    pinch_color = (0, 0, 255) if pinch_distance < self.pinch_threshold_3d else (255, 255, 255)
                    cv2.circle(frame, (10, 140), 10, pinch_color, -1)
                      # Show hand position for monitor switching
                    middle_finger_control = hand_landmarks.landmark[9]
                    norm_x = (middle_finger_control.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
                    norm_x = max(0, min(1, norm_x))
                    
                    hand_pos_text = f"Control X: {norm_x:.2f}"
                    cv2.putText(frame, hand_pos_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # Show zones based on normalized position
                    if norm_x < 0.1:
                        cv2.putText(frame, "LEFT ZONE", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif norm_x > 0.9:
                        cv2.putText(frame, "RIGHT ZONE", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "CENTER", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Simple window title like the original
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracking = HandTrackingMouseControlSimpleDisplay()
    try:
        hand_tracking.run()
    except KeyboardInterrupt:
        print("Script has been terminated!")
        pass