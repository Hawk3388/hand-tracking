import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import os
import sys
import math
import numpy as np
from pathlib import Path
from screeninfo import get_monitors

# ASL modules as global variables
torch = None
FrameMLP2 = None
ASL_AVAILABLE = True

class HandTrackingMouseControl:
    def __init__(self, enable_asl=True):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Works in Windows & Linux/Mac
        time.sleep(0.001)
        os.system("cls" if os.name == "nt" else "clear")
        print("Hand Tracking with ASL Recognition" if enable_asl and ASL_AVAILABLE else "Hand Tracking - Mouse Control")
        print("press ctrl + c to exit")
        
        # ASL Setup
        self.enable_asl = enable_asl and ASL_AVAILABLE
        self.asl_model = None
        if self.enable_asl:
            self._setup_asl()
        else:
            self._setup_asl()
        
        # Mode switch setup
        self.asl_mode = False  # False = mouse, True = ASL keyboard input
        self.mode_switch_cooldown = 0
        self.mode_switch_delay = 1.5  # 1.5 second cooldown
        self.mode_switching = False  # Mode switch status
        
        # ASL keyboard input setup
        self.last_typed_letter = ""
        self.letter_cooldown = 0
        self.letter_delay = 1.0  # 1 Sekunde zwischen Buchstaben

        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is available
        if not self.cap.isOpened():
            print("Error: Camera could not be opened!")
            raise Exception("Camera not available")
        
        # Multi-Monitor Setup
        self.setup_multi_monitor()
        print(f"Detected monitors: {len(self.monitors)}")
        for i, monitor in enumerate(self.monitors):
            print(f"  Monitor {i+1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
        
        self.camera_width, self.camera_height = 640, 480
        
        # Define regions for different monitors/functions
        self.region_x_min, self.region_x_max = 0.2, 0.8
        self.region_y_min, self.region_y_max = 0.2, 0.8
        
        # Monitor switching parameters
        self.current_monitor = 0
        self.last_monitor_switch = 0
        self.monitor_switch_cooldown = 1.0  # Cooldown between monitor switches
        self.monitor_switch_zone_width = 0.15  # 15% zones left/right for monitor switching
        
        # 3D tracking parameters (more sensitive)
        self.z_baseline = None
        self.z_smoothing_factor = 0.8  # Higher for more stability
        self.smoothed_z = 0
        self.depth_scale = 100  # Scaling factor for depth display
        
        # Click detection (pure 3D)
        self.clicking = False
        self.click_start_time = None
        self.holding = False
        self.mouse_position = (0, 0)
        self.mouse_action = None
        self.last_gesture_time = 0
        self.gesture_delay = 0.05  # Faster reaction
        
        # Position smoothing
        self.position_history = []
        self.position_smoothing = 3  # Less smoothing for more responsiveness
        
        # 3D distance for click (optimized for 3D)
        self.pinch_threshold_3d = 0.04  # Very narrow threshold for precise 3D control
        self.pinch_release_threshold_3d = 0.06
        
        # Disable PyAutoGUI Fail-Safe (use at your own risk!)
        pyautogui.FAILSAFE = False
        
        # Start the separate thread for mouse control
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
        
        # Start monitor border thread
        self.monitor_border_thread = threading.Thread(target=self.draw_monitor_borders, daemon=True)
        self.monitor_border_thread.start()
        
        # Initialize control region based on current monitor
        self.update_control_region_for_monitor()
    
    def _setup_asl(self):
        """Setup for ASL recognition - EXACTLY LIKE TEST_CAMERA"""
        # ASL Import
        global torch, FrameMLP2
        try:
            asl_path = str(Path(__file__).parent.parent / "asl")
            if asl_path not in sys.path:
                sys.path.insert(0, asl_path)
            
            from model import FrameMLP2 as _FrameMLP2
            import torch as _torch
            torch = _torch
            FrameMLP2 = _FrameMLP2
        except Exception as e:
            print(f"Error importing ASL: {e}")
            self.enable_asl = False
            return
            
        model_path = Path(__file__).parent.parent / "asl" / "model" / "frame_mlp_asl.pt"
        
        if not model_path.exists():
            print(f"ASL model not found: {model_path}")
            self.enable_asl = False
            return
        
        try:
            device = torch.device("cpu")
            self.device = device
            asl_model = FrameMLP2()
            asl_model.load_state_dict(torch.load(model_path, map_location=device))
            asl_model.to(device)
            asl_model.eval()
            self.asl_model = asl_model
            
            # Only 26 letters (A-Z) as trained in the model
            self.class_names = [chr(ord('A') + i) for i in range(26)]
            
            self.prediction_history = []
            self.current_prediction = ""
            self.prediction_confidence = 0.0
            
            print(f"ASL model loaded! (Device: {self.device})")
        except Exception as e:
            print(f"Error loading ASL model: {e}")
            self.enable_asl = False
    
    def _extract_landmarks(self, hand_landmarks):
        """Extracts hand landmarks for ASL"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def _predict_asl(self, landmarks):
        """EXACTLY LIKE TEST_CAMERA with debug"""
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.asl_model(x)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
        return self.class_names[pred_idx], confidence
    
    def _smooth_asl_prediction(self, letter, confidence):
        """EXACTLY like in test_camera.py"""
        self.prediction_history.append((letter, confidence))
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Most common letter with high confidence
        letter_counts = {}
        total_confidence = 0
        
        for l, c in self.prediction_history[-5:]:  # Only last 5 frames
            if c > 0.7:  # Only high confidence
                letter_counts[l] = letter_counts.get(l, 0) + 1
                total_confidence += c
        
        if letter_counts:
            most_common = max(letter_counts.items(), key=lambda x: x[1])
            avg_conf = total_confidence / len(self.prediction_history[-5:])
            return most_common[0], avg_conf
        
        return letter, confidence
    
    def setup_multi_monitor(self):
        """Set up multi-monitor configuration."""
        raw_monitors = list(get_monitors())
        
        # Sort monitors by X position (left to right) for consistent order
        self.monitors = sorted(raw_monitors, key=lambda m: m.x)
        
        # Calculate virtual desktop boundaries correctly
        min_x = min(m.x for m in self.monitors)
        min_y = min(m.y for m in self.monitors)
        max_x = max(m.x + m.width for m in self.monitors)
        max_y = max(m.y + m.height for m in self.monitors)
        
        self.total_screen_width = max_x - min_x
        self.total_screen_height = max_y - min_y
        self.virtual_desktop_offset_x = min_x
        self.virtual_desktop_offset_y = min_y
        
        # Find primary monitor index
        self.primary_monitor_index = 0
        for i, monitor in enumerate(self.monitors):
            if hasattr(monitor, 'is_primary') and monitor.is_primary:
                self.primary_monitor_index = i
                break
        
        self.primary_monitor = self.monitors[self.primary_monitor_index]
    
    def detect_monitor_zone_switch(self, hand_landmarks):
        """Detect monitor switch based on hand position in adaptive edge zones."""
        current_time = time.time()
        
        if current_time - self.last_monitor_switch < self.monitor_switch_cooldown:
            return
        
        if len(self.monitors) <= 1:
            return
        
        # Use the SAME point that controls the mouse (middle finger MCP joint 9)
        middle_finger_control = hand_landmarks.landmark[9]
        hand_x = middle_finger_control.x
        
        # Calculate adaptive zone boundaries based on current control region
        gap_width = 0.05  # 5% gap between green control area and red switch zones
        
        # Left zone: from left edge to before green control region (with gap)
        left_zone_end = max(0, self.region_x_min - gap_width)
        
        # Right zone: from after green control region (with gap) to right edge
        right_zone_start = min(1.0, self.region_x_max + gap_width)
        
        # Check if hand is in left monitor switch zone
        if hand_x < left_zone_end and left_zone_end > 0.05:  # Only if zone has reasonable width
            # Try to switch to left monitor
            current_monitor_obj = self.monitors[self.current_monitor]
            target_monitor_index = None
            
            # Find monitor to the left (adjacent)
            for i, monitor in enumerate(self.monitors):
                if (monitor.x + monitor.width <= current_monitor_obj.x and 
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if target_monitor_index is None or monitor.x > self.monitors[target_monitor_index].x:
                        target_monitor_index = i
            
            if target_monitor_index is not None:
                self.current_monitor = target_monitor_index
                self.last_monitor_switch = current_time
                print(f"🖱️ LEFT ZONE → Monitor {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
        
        # Check if hand is in right monitor switch zone
        elif hand_x > right_zone_start and (1.0 - right_zone_start) > 0.05:  # Only if zone has reasonable width
            # Try to switch to right monitor
            current_monitor_obj = self.monitors[self.current_monitor]
            target_monitor_index = None
            
            # Find monitor to the right (adjacent)
            for i, monitor in enumerate(self.monitors):
                if (monitor.x >= current_monitor_obj.x + current_monitor_obj.width and
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if target_monitor_index is None or monitor.x < self.monitors[target_monitor_index].x:
                        target_monitor_index = i
            
            if target_monitor_index is not None:
                self.current_monitor = target_monitor_index
                self.last_monitor_switch = current_time
                print(f"🖱️ RIGHT ZONE → Monitor {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
    
    def convert_to_monitor_coordinates(self, norm_x, norm_y, monitor_index):
        """Convert normalized coordinates to specific monitor coordinates."""
        if monitor_index >= len(self.monitors):
            monitor_index = 0
        
        monitor = self.monitors[monitor_index]
        
        # Calculate absolute coordinates within specific monitor
        local_x = norm_x * monitor.width
        local_y = norm_y * monitor.height
        
        # Add monitor offset to get global coordinates
        global_x = int(local_x + monitor.x)
        global_y = int(local_y + monitor.y)
        
        return global_x, global_y
    
    def calculate_3d_distance(self, point1, point2):
        """Calculate 3D Euclidean distance between two landmark points."""
        return math.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2
        )
    
    def smooth_position(self, new_position):
        """Smooth mouse position with moving average."""
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
        
        elif pinch_distance_3d > self.pinch_release_threshold_3d:
            if self.clicking and self.holding:
                pyautogui.mouseUp()
                self.clicking = False
                self.holding = False
                self.last_gesture_time = current_time
    
    def draw_monitor_borders(self):
        """Draw permanent borders around current active monitor."""
        try:
            import tkinter as tk
            
            current_windows = []
            last_monitor = -1
            
            while True:
                # Check if monitor switched or borders need to be created
                if self.current_monitor != last_monitor:
                    # First clean up old windows
                    for window in current_windows:
                        try:
                            window.destroy()
                        except:
                            pass
                    current_windows = []
                    
                    # Create new borders for current monitor
                    monitor = self.monitors[self.current_monitor]
                    
                    try:
                        # Create invisible root window
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-alpha', 0.0)
                        
                        border_width = 8  # Increased border width for better visibility
                        border_color = '#0080FF'  # Light blue border
                        
                        # Calculate border positions
                        top_x, top_y = monitor.x, monitor.y
                        bottom_x, bottom_y = monitor.x, monitor.y + monitor.height - border_width
                        left_x, left_y = monitor.x, monitor.y
                        right_x, right_y = monitor.x + monitor.width - border_width, monitor.y
                        
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
                        
                    except Exception as e:
                        pass
                
                # Keep borders active and prevent flickering
                try:
                    if current_windows and current_windows[-1]:  # root window
                        current_windows[-1].update()
                except:
                    pass
                
                time.sleep(0.1)
        except Exception as e:
            time.sleep(1)
    
    def update_control_region_for_monitor(self):
        """Update control region based on actual pixel size and orientation of current monitor."""
        if not self.monitors or self.current_monitor >= len(self.monitors):
            if self.monitors:
                self.current_monitor = 0  # Reset to first monitor
            else:
                return
        
        current_monitor = self.monitors[self.current_monitor]
        monitor_width = current_monitor.width
        monitor_height = current_monitor.height
        monitor_aspect_ratio = monitor_width / monitor_height
        
        # Camera resolution (fixed)
        camera_aspect_ratio = self.camera_width / self.camera_height
        
        # Fixed margin from camera edge (in normalized coordinates)
        fixed_margin = 0.08  # 8% margin on all sides
        
        # Additional margin for monitor switch zones if multiple monitors exist
        extra_horizontal_margin = 0.0
        if len(self.monitors) > 1:
            extra_horizontal_margin = self.monitor_switch_zone_width  # Reserve space for monitor switch zones
        
        # Available space for control region
        available_width = 1.0 - 2 * (fixed_margin + extra_horizontal_margin)
        available_height = 1.0 - 2 * fixed_margin
        
        # Calculate largest possible rectangle with monitor aspect ratio
        if monitor_aspect_ratio > (available_width / available_height):
            # Monitor is relatively wider than available space - width is limiting factor
            region_width = available_width
            region_height = region_width / monitor_aspect_ratio
        else:
            # Monitor is relatively taller than available space - height is limiting factor
            region_height = available_height
            region_width = region_height * monitor_aspect_ratio
        
        # Center the region in camera image
        horizontal_center = 0.5
        vertical_center = 0.5
        
        self.region_x_min = horizontal_center - region_width / 2
        self.region_x_max = horizontal_center + region_width / 2
        self.region_y_min = vertical_center - region_height / 2
        self.region_y_max = vertical_center + region_height / 2

    def handle_mouse(self):
        """Function to handle mouse actions in a separate thread."""
        while True:
            if self.mouse_action == "move":
                pyautogui.moveTo(*self.mouse_position)
            self.mouse_action = None
            time.sleep(0.001)  # Reduce CPU usage
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)  # Flip for natural control
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))  # Ensure size matches
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create black background instead of camera image
            frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Only process the first detected hand
                
                # Analyze hand depth for 3D tracking
                self.analyze_hand_depth(hand_landmarks)
                
                # Mode switch gesture (EXACTLY like thumb+index finger click!)
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                middle_finger = hand_landmarks.landmark[12]  # Middle finger tip
                distance = abs(thumb_tip.x - middle_finger.x) + abs(thumb_tip.y - middle_finger.y)
                
                if distance <= 0.05:  # When thumb and middle finger touch
                    if not self.mode_switching:
                        self.mode_switching = True
                        self.asl_mode = not self.asl_mode
                        mode_name = "ASL Keyboard" if self.asl_mode else "Mouse Control"
                        print(f"Mode switched to: {mode_name}")
                elif distance > 0.05:
                    if self.mode_switching:
                        self.mode_switching = False
                
                # ASL recognition (only in ASL mode)
                if self.enable_asl and self.asl_mode:
                    # Extract landmarks and predict
                    landmarks = self._extract_landmarks(hand_landmarks)
                    letter, confidence = self._predict_asl(landmarks)
                    
                    # Smooth prediction
                    smooth_letter, smooth_confidence = self._smooth_asl_prediction(letter, confidence)
                    
                    self.current_prediction = smooth_letter
                    self.prediction_confidence = smooth_confidence
                    
                    # Type at high confidence
                    if smooth_confidence > 0.8:
                        if not hasattr(self, 'last_letter_time'):
                            self.last_letter_time = 0
                        if not hasattr(self, 'last_letter'):
                            self.last_letter = ""
                            
                        current_time = time.time()
                        if current_time - self.last_letter_time > 1.0 and smooth_letter != self.last_letter:
                            pyautogui.write(smooth_letter.lower())
                            print(f"Typed: {smooth_letter}")
                            self.last_letter_time = current_time
                            self.last_letter = smooth_letter
                
                # Mouse control (only in mouse mode)
                if not self.asl_mode:
                    # Use middle finger MCP (joint 9) for more stable control
                    middle_finger_control = hand_landmarks.landmark[9]
                    
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
                    
                    # Gesture recognition
                    self.detect_monitor_zone_switch(hand_landmarks)
                    self.detect_3d_click(hand_landmarks)
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Display mode and control
            mode_text = "ASL Keyboard Mode" if self.asl_mode else "Mouse Mode"
            mode_color = (0, 255, 0) if self.asl_mode else (0, 255, 255)
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            cv2.putText(frame, "Thumb+Middle Finger = Switch Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display ASL info (only in ASL mode)
            if self.enable_asl and self.asl_mode:
                y_offset = 90
                if hasattr(self, 'current_prediction'):
                    confidence_text = f"({self.prediction_confidence:.2f})" if hasattr(self, 'prediction_confidence') else ""
                    cv2.putText(frame, f"Letter: {self.current_prediction} {confidence_text}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Display typing buffer
                if hasattr(self, 'typing_buffer') and self.typing_buffer:
                    cv2.putText(frame, f"Buffer: {self.typing_buffer}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                # Gesture help
                cv2.putText(frame, "Thumb+Pinky = Space", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, "Index+Ring = Backspace", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, "All fingers together = Enter", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Confidence: {self.prediction_confidence:.2f}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Control hints
            cv2.putText(frame, "Thumb+Middle Finger = Switch Mode | M = Switch Mode | S = Space", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw control region and multi-monitor information
            region_start_x = int(self.region_x_min * self.camera_width)
            region_end_x = int(self.region_x_max * self.camera_width)
            region_start_y = int(self.region_y_min * self.camera_height)
            region_end_y = int(self.region_y_max * self.camera_height)
            cv2.rectangle(frame, (region_start_x, region_start_y), 
                         (region_end_x, region_end_y), (0, 255, 0), 2)
            
            # Draw monitor switch zones (left and right edges)
            if len(self.monitors) > 1 and not self.asl_mode:
                gap_width = 0.05  # 5% gap between green control area and red switch zones
                
                # Left zone
                left_zone_end = max(0, self.region_x_min - gap_width)
                left_zone_width = left_zone_end - 0
                
                # Right zone
                right_zone_start = min(1.0, self.region_x_max + gap_width)
                right_zone_width = 1.0 - right_zone_start
                
                # Convert to pixel coordinates
                left_zone_end_px = int(left_zone_end * self.camera_width)
                right_zone_start_px = int(right_zone_start * self.camera_width)
                
                # Draw left zone (only if it has reasonable width)
                if left_zone_width > 0.05:  # At least 5% width
                    cv2.rectangle(frame, (0, 0), (left_zone_end_px, self.camera_height), (0, 0, 255), 2)

                # Draw right zone (only if it has reasonable width)
                if right_zone_width > 0.05:  # At least 5% width
                    cv2.rectangle(frame, (right_zone_start_px, 0), (self.camera_width, self.camera_height), (0, 0, 255), 2)
            
            # Show multi-monitor information (only in mouse mode)
            if not self.asl_mode and len(self.monitors) > 1:
                # Show current monitor information
                monitor_text = f"Monitor: {self.current_monitor + 1}/{len(self.monitors)}"
                cv2.putText(frame, monitor_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show 3D depth and click information
                if hasattr(self, 'smoothed_z'):
                    depth_text = f"Depth: {self.smoothed_z:.3f}"
                    cv2.putText(frame, depth_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Visual depth display
                    depth_bar_length = int(abs(self.smoothed_z) * self.depth_scale)
                    depth_bar_length = min(200, max(0, depth_bar_length))
                    color = (0, 255, 0) if self.smoothed_z < 0 else (0, 0, 255)
                    cv2.rectangle(frame, (10, 140), (10 + depth_bar_length, 150), color, -1)
                
                # Show 3D click distance when hand is detected
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    pinch_distance = self.calculate_3d_distance(thumb_tip, index_tip)
                    
                    pinch_text = f"3D Click: {pinch_distance:.3f}"
                    cv2.putText(frame, pinch_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Visual click display
                    pinch_color = (0, 0, 255) if pinch_distance < self.pinch_threshold_3d else (255, 255, 255)
                    cv2.circle(frame, (10, 190), 8, pinch_color, -1)
            
            cv2.imshow("Hand Tracking", frame)
            
            # Process keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):  # 'M' key for mode switch
                self.asl_mode = not self.asl_mode
                mode_name = "ASL Keyboard" if self.asl_mode else "Mouse Control"
                print(f"Mode switched to: {mode_name}")
            elif key == ord('s') and self.asl_mode:  # 'S' key for space
                pyautogui.write(' ')
                print("Space typed")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand Tracking with ASL recognition")
    parser.add_argument("--no-asl", action="store_true", help="Disable ASL recognition")
    args = parser.parse_args()
    
    hand_tracking = HandTrackingMouseControl(enable_asl=not args.no_asl)
    try:
        hand_tracking.run()
    except KeyboardInterrupt:
        print("Script has been terminated!")
        pass
