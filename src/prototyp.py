import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import os
import math
from screeninfo import get_monitors

class HandTracking:
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
        print("Press Ctrl or Ctrl + C to exit")
        print("Move your hand to the red edges of the camera image to switch between monitors")

        self.cap = cv2.VideoCapture(0)
        
        # Check if camera is available
        if not self.cap.isOpened():
            print("Error: Camera could not be opened!")
            raise Exception("Camera not available")        # Multi-Monitor Setup
        self.setup_multi_monitor()
        print(f"Detected monitors: {len(self.monitors)}")
        for i, monitor in enumerate(self.monitors):
            print(f"  Monitor {i+1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
            if hasattr(monitor, 'is_primary'):
                print(f"    Primary: {monitor.is_primary}")
        print(f"Total screen size: {self.total_screen_width}x{self.total_screen_height}")
        
        self.camera_width, self.camera_height = 640, 480
        
        # Define regions for different monitors/functions
        self.region_x_min, self.region_x_max = 0.2, 0.8
        self.region_y_min, self.region_y_max = 0.2, 0.8
        # Monitor switching parameters
        self.current_monitor = 0
        # Monitor switching with edge zones
        self.last_monitor_switch = 0
        self.monitor_switch_cooldown = 1.0  # Cooldown between monitor switches
        self.monitor_switch_zone_width = 0.15  # 15% zones left/right for monitor switching
        # Direction switching
        self.directional_switch_active = False
        self.pointing_threshold = 0.07
        self.pointing_depth_threshold = 0.06
        self.last_pointing_switch = 0
        self.pointing_switch_cooldown = 0.5  # Schnellere Schaltung
        
        # Monitorzonen für Richtungsschaltung
        self.monitor_zones = self.setup_monitor_zones()
        
        # Visuelles Feedback
        self.monitor_highlight_duration = 2.0
        self.monitor_highlight_start = 0
        self.highlighted_monitor = -1
        # 3D-Tracking-Parameter (empfindlicher)
        self.z_baseline = None
        self.z_smoothing_factor = 0.8  # Höher für mehr Stabilität
        self.smoothed_z = 0
        self.depth_scale = 100  # Skalierungsfaktor für Tiefenanzeige
        # Klick-Erkennung (rein 3D)
        self.clicking = False
        self.click_start_time = None
        self.holding = False
        self.last_gesture_time = 0
        self.gesture_delay = 0.05  # Schnellere Reaktion
        
        # Positionsglättung
        self.position_history = []
        self.position_smoothing = 3  # Weniger Glättung für mehr Reaktionsfähigkeit
        
        # 3D-Abstand für Klick (optimiert für 3D)
        self.pinch_threshold_3d = 0.04  # Sehr enger Schwellenwert für präzise 3D-Steuerung
        self.pinch_release_threshold_3d = 0.06
        
        pyautogui.FAILSAFE = False
        
        # Starte Threads
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
        
        self.monitor_border_thread = threading.Thread(target=self.draw_monitor_borders, daemon=True)
        self.monitor_border_thread.start()
        
        # Initialisiere Steuerungsregion basierend auf aktuellem Monitor
        self.update_control_region_for_monitor()
        
    def setup_multi_monitor(self):
        """Einrichten der Multi-Monitor-Konfiguration mit umfassendem Debugging."""

        raw_monitors = list(get_monitors())
        
        print("=== ROHE MONITORDETEKTION ===")
        for i, monitor in enumerate(raw_monitors):
            print(f"  Rohmonitor {i}: {monitor}")
            print(f"    Größe: {monitor.width}x{monitor.height}")
            print(f"    Position: ({monitor.x}, {monitor.y})")
            if hasattr(monitor, 'is_primary'):
                print(f"    Primär: {monitor.is_primary}")
            if hasattr(monitor, 'name'):
                print(f"    Name: {monitor.name}")
        
        # Sortiere Monitore nach X-Position (von links nach rechts) für konsistente Reihenfolge
        self.monitors = sorted(raw_monitors, key=lambda m: m.x)
        
        print("\n=== SORTIERTE MONITORE (von links nach rechts) ===")
        for i, monitor in enumerate(self.monitors):
            print(f"  Monitor {i}: Größe {monitor.width}x{monitor.height} bei ({monitor.x}, {monitor.y})")
            if hasattr(monitor, 'is_primary'):
                print(f"    Primär: {monitor.is_primary}")
            if hasattr(monitor, 'name'):
                print(f"    Name: {monitor.name}")
        
        # Berechne virtuelle Desktop-Grenzen korrekt
        min_x = min(m.x for m in self.monitors)
        min_y = min(m.y for m in self.monitors)
        max_x = max(m.x + m.width for m in self.monitors)
        max_y = max(m.y + m.height for m in self.monitors)
        
        self.total_screen_width = max_x - min_x
        self.total_screen_height = max_y - min_y
        self.virtual_desktop_offset_x = min_x
        self.virtual_desktop_offset_y = min_y
        
        print(f"\n=== VIRTUELLE DESKTOP-BERECHNUNG ===")
        print(f"  Min-Grenzen: ({min_x}, {min_y})")
        print(f"  Max-Grenzen: ({max_x}, {max_y})")
        print(f"  Gesamtgröße: {self.total_screen_width}x{self.total_screen_height}")
        print(f"  Virtueller Offset: ({self.virtual_desktop_offset_x}, {self.virtual_desktop_offset_y})")
        
        # Finde primären Monitor-Index
        self.primary_monitor_index = 0
        for i, monitor in enumerate(self.monitors):
            if hasattr(monitor, 'is_primary') and monitor.is_primary:
                self.primary_monitor_index = i
                break
        
        self.primary_monitor = self.monitors[self.primary_monitor_index]
        print(f"\n=== PRIMÄRMONITOR ===")
        print(f"  Primärer Monitor-Index: {self.primary_monitor_index}")
        print(f"  Primärer Monitor: {self.primary_monitor.width}x{self.primary_monitor.height} bei ({self.primary_monitor.x}, {self.primary_monitor.y})")
    
    def setup_monitor_zones(self):
        """Einrichten von Zonen für die Richtungsschaltung von Monitoren."""
        zones = []
        if len(self.monitors) <= 1:
            return zones
        
        # Sortiere Monitore nach X-Position (von links nach rechts)
        sorted_monitors = sorted(enumerate(self.monitors), key=lambda x: x[1].x)
        
        # Erstelle Richtungzonen
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
    
    def detect_monitor_zone_switch(self, hand_landmarks):
        """Erkenne den Monitorwechsel basierend auf der Handposition in adaptiven Randzonen."""
        current_time = time.time()
        
        if current_time - self.last_monitor_switch < self.monitor_switch_cooldown:
            return
        
        if len(self.monitors) <= 1:
            return
        
        # Verwende den GLEICHEN Punkt, der die Maus steuert (Mittelhand-MCP-Gelenk 9)
        # Dies gewährleistet ein konsistentes Verhalten zwischen Maussteuerung und Monitorwechsel
        middle_finger_control = hand_landmarks.landmark[9]
        hand_x = middle_finger_control.x
          # Berechne adaptive Zonen-Grenzen basierend auf der aktuellen Steuerungsregion
        gap_width = 0.05  # 5% Lücke zwischen grünem Steuerbereich und roten Wechselzonen
        
        # Linke Zone: von der linken Kante bis vor die grüne Steuerregion (mit Lücke)
        left_zone_end = max(0, self.region_x_min - gap_width)
        
        # Rechte Zone: von nach der grünen Steuerregion (mit Lücke) bis zur rechten Kante
        right_zone_start = min(1.0, self.region_x_max + gap_width)
        
        # Überprüfe, ob die Hand in der linken Monitorschaltzone ist
        if hand_x < left_zone_end and left_zone_end > 0.05:  # Nur wenn die Zone eine angemessene Breite hat
            # Versuche, zum linken Monitor zu wechseln
            current_monitor_obj = self.monitors[self.current_monitor]
            target_monitor_index = None
            
            # Finde Monitor links (benachbart)
            for i, monitor in enumerate(self.monitors):
                if (monitor.x + monitor.width <= current_monitor_obj.x and 
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if target_monitor_index is None or monitor.x > self.monitors[target_monitor_index].x:
                        target_monitor_index = i
            
            if target_monitor_index is not None:
                self.current_monitor = target_monitor_index
                self.last_monitor_switch = current_time
                print(f"🖱️ ZONE LINKS → Monitor {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
        
        # Überprüfe, ob die Hand in der rechten Monitorschaltzone ist
        elif hand_x > right_zone_start and (1.0 - right_zone_start) > 0.05:  # Nur wenn die Zone eine angemessene Breite hat
            # Versuche, zum rechten Monitor zu wechseln
            current_monitor_obj = self.monitors[self.current_monitor]
            target_monitor_index = None
            
            # Finde Monitor rechts (benachbart)
            for i, monitor in enumerate(self.monitors):
                if (monitor.x >= current_monitor_obj.x + current_monitor_obj.width and
                    monitor.y < current_monitor_obj.y + current_monitor_obj.height and
                    monitor.y + monitor.height > current_monitor_obj.y):
                    if target_monitor_index is None or monitor.x < self.monitors[target_monitor_index].x:
                        target_monitor_index = i
            
            if target_monitor_index is not None:
                self.current_monitor = target_monitor_index
                self.last_monitor_switch = current_time
                print(f"🖱️ ZONE RECHTS → Monitor {self.current_monitor + 1}")
                self.update_control_region_for_monitor()
    
    def get_monitor_from_pointing_direction(self, hand_x):
        """Bestimme den Zielmonitor basierend auf der Zeigerrichtung."""
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
        """Zeichne permanente Ränder um den aktuellen aktiven Monitor mit verbessertem Debugging."""
        try:
            import tkinter as tk
            
            current_windows = []
            last_monitor = -1
            
            while True:
                # Überprüfe, ob der Monitor gewechselt wurde oder ob Ränder erstellt werden müssen
                if self.current_monitor != last_monitor:
                    # Zuerst alte Fenster bereinigen
                    for window in current_windows:
                        try:
                            window.destroy()
                        except:
                            pass
                    current_windows = []
                    
                    # Neue Ränder für den aktuellen Monitor erstellen
                    monitor = self.monitors[self.current_monitor]
                    print(f"\n>>> ERSTELLE RAND AM MONITOR {self.current_monitor + 1} <<<")
                    print(f"  Monitorgröße: {monitor.width}x{monitor.height}")
                    print(f"  Monitorposition: ({monitor.x}, {monitor.y})")
                    print(f"  Rand deckt Bereich ab: ({monitor.x}, {monitor.y}) bis ({monitor.x + monitor.width}, {monitor.y + monitor.height})")
                    
                    try:
                        # Unsichtbares Root-Fenster erstellen
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-alpha', 0.0)
                        
                        border_width = 8  # Erhöhte Randbreite für bessere Sichtbarkeit
                        border_color = '#0080FF'  # Heller blauer Rand
                        
                        # Berechne Randpositionen
                        top_x, top_y = monitor.x, monitor.y
                        bottom_x, bottom_y = monitor.x, monitor.y + monitor.height - border_width
                        left_x, left_y = monitor.x, monitor.y
                        right_x, right_y = monitor.x + monitor.width - border_width, monitor.y
                        
                        print(f"  Randpositionen:")
                        print(f"    Oben: {monitor.width}x{border_width} bei ({top_x}, {top_y})")
                        print(f"    Unten: {monitor.width}x{border_width} bei ({bottom_x}, {bottom_y})")
                        print(f"    Links: {border_width}x{monitor.height} bei ({left_x}, {left_y})")
                        print(f"    Rechts: {border_width}x{monitor.height} bei ({right_x}, {right_y})")
                        
                        # Oben Rand
                        top = tk.Toplevel(root)
                        top.geometry(f"{monitor.width}x{border_width}+{top_x}+{top_y}")
                        top.configure(bg=border_color)
                        top.overrideredirect(True)
                        top.attributes('-topmost', True)
                        top.attributes('-alpha', 0.8)
                        current_windows.append(top)
                        
                        # Unten Rand
                        bottom = tk.Toplevel(root)
                        bottom.geometry(f"{monitor.width}x{border_width}+{bottom_x}+{bottom_y}")
                        bottom.configure(bg=border_color)
                        bottom.overrideredirect(True)
                        bottom.attributes('-topmost', True)
                        bottom.attributes('-alpha', 0.8)
                        current_windows.append(bottom)
                        
                        # Links Rand
                        left = tk.Toplevel(root)
                        left.geometry(f"{border_width}x{monitor.height}+{left_x}+{left_y}")
                        left.configure(bg=border_color)
                        left.overrideredirect(True)
                        left.attributes('-topmost', True)
                        left.attributes('-alpha', 0.8)
                        current_windows.append(left)
                        
                        # Rechts Rand
                        right = tk.Toplevel(root)
                        right.geometry(f"{border_width}x{monitor.height}+{right_x}+{right_y}")
                        right.configure(bg=border_color)
                        right.overrideredirect(True)
                        right.attributes('-topmost', True)
                        right.attributes('-alpha', 0.8)
                        current_windows.append(right)
                        
                        current_windows.append(root)
                        last_monitor = self.current_monitor
                        print(f"  ✓ Randfenster erfolgreich erstellt")
                        
                    except Exception as e:
                        print(f"  ✗ Rand-Erstellungsfehler: {e}")
                
                # Halte Ränder aktiv und verhindere Flackern
                try:
                    if current_windows and current_windows[-1]:  # root-Fenster
                        current_windows[-1].update()
                except:
                    pass
                
                time.sleep(0.1)
        except Exception as e:
            print(f"Rand-Thread-Fehler: {e}")
            time.sleep(1)
    
    def convert_to_monitor_coordinates(self, norm_x, norm_y, monitor_index):
        """Konvertiere normalisierte Koordinaten in spezifische Monitor-Koordinaten mit umfassendem Debugging."""
        if monitor_index >= len(self.monitors):
            print(f"WARNUNG: Ungültiger Monitor-Index {monitor_index}, verwende 0")
            monitor_index = 0
        
        monitor = self.monitors[monitor_index]
        
        # Berechne absolute Koordinaten innerhalb des spezifischen Monitors
        local_x = norm_x * monitor.width
        local_y = norm_y * monitor.height
        
        # Füge Monitor-Offset hinzu, um globale Koordinaten zu erhalten
        global_x = int(local_x + monitor.x)
        global_y = int(local_y + monitor.y)
        
        # Debug-Ausgabe für Koordinaten
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        # Drucke Debug-Info alle 60 Frames (2 Sekunden bei 30fps), um Spam zu vermeiden
        if self.debug_counter % 60 == 0:
            print(f"\n=== KOORDINATEN-UMRECHNUNG-DESKTOP ===")
            print(f"  Zielmonitor {monitor_index}: {monitor.width}x{monitor.height} bei ({monitor.x}, {monitor.y})")
            print(f"  Normalisierte Eingabe: ({norm_x:.3f}, {norm_y:.3f})")
            print(f"  Lokale Koordinaten: ({local_x:.1f}, {local_y:.1f})")
            print(f"  Globale Koordinaten: ({global_x}, {global_y})")
            
            # Überprüfe, ob die Koordinaten innerhalb der Monitorgrenzen liegen
            if monitor.x <= global_x <= monitor.x + monitor.width and monitor.y <= global_y <= monitor.y + monitor.height:
                print(f"  ✓ Koordinaten liegen innerhalb der Monitorgrenzen")
            else:
                print(f"  ⚠ Koordinaten liegen AUßERHALB der Monitorgrenzen!")
                print(f"    Monitorgrenzen: x=[{monitor.x}, {monitor.x + monitor.width}], y=[{monitor.y}, {monitor.y + monitor.height}]")
        
        return global_x, global_y
    
    def calculate_3d_distance(self, point1, point2):
        """Berechne die 3D-Euklidische Distanz zwischen zwei Markierungspunkten."""
        return math.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2 + 
            (point1.z - point2.z) ** 2
        )
    
    def smooth_position(self, new_position):
        """Glätte die Mausposition mit einem gleitenden Durchschnitt."""
        self.position_history.append(new_position)
        if len(self.position_history) > self.position_smoothing:
            self.position_history.pop(0)
        
        avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        return (int(avg_x), int(avg_y))
    
    def analyze_hand_depth(self, hand_landmarks):
        """Analysiere die Handtiefe."""
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
        """Reine 3D-Klickerkennung nur mit 3D-Distanz zwischen Daumen und Zeigefinger."""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_delay:
            return
        
        if not self.directional_switch_active:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
              # Berechne nur die 3D-Distanz
            pinch_distance_3d = self.calculate_3d_distance(thumb_tip, index_tip)
            
            if pinch_distance_3d < self.pinch_threshold_3d:
                if not self.clicking:
                    self.clicking = True
                    self.holding = True
                    pyautogui.mouseDown()
                    self.last_gesture_time = current_time
                    print(f"3D Klick! Distanz: {pinch_distance_3d:.3f}")
            
            elif pinch_distance_3d > self.pinch_release_threshold_3d:
                if self.clicking and self.holding:
                    pyautogui.mouseUp()
                    self.clicking = False
                    self.holding = False
                    self.last_gesture_time = current_time
                    print(f"3D Freigabe! Distanz: {pinch_distance_3d:.3f}")

    def handle_mouse(self):
        """Funktion zur Handhabung von Mausaktionen in einem separaten Thread."""
        while True:
            if hasattr(self, 'mouse_position') and hasattr(self, 'mouse_action'):
                if self.mouse_action == "move":
                    pyautogui.moveTo(*self.mouse_position)
                self.mouse_action = None
            time.sleep(0.001)
    
    def update_control_region_for_monitor(self):
        """Aktualisiere die Steuerungsregion basierend auf der tatsächlichen Pixelgröße und Ausrichtung des aktuellen Monitors."""
        if not self.monitors or self.current_monitor >= len(self.monitors):
            print(f"WARNUNG: Ungültige Monitor-Konfiguration. Aktueller Monitor: {self.current_monitor}, Verfügbare Monitore: {len(self.monitors)}")
            if self.monitors:
                self.current_monitor = 0  # Setze zurück auf ersten Monitor
                print(f"Zurückgesetzt auf Monitor 0")
            else:
                print("Keine Monitore verfügbar, überspringe Aktualisierung der Steuerungsregion")
                return
        
        current_monitor = self.monitors[self.current_monitor]
        monitor_width = current_monitor.width
        monitor_height = current_monitor.height
        monitor_aspect_ratio = monitor_width / monitor_height
        
        print(f"\n=== STEUERUNGSREGION AKTUALISIEREN ===")
        print(f"  Aktueller Monitor: {monitor_width}x{monitor_height}")
        print(f"  Monitor-Seitenverhältnis: {monitor_aspect_ratio:.3f}")
        
        # Kamerarauflösung (fest)
        camera_aspect_ratio = self.camera_width / self.camera_height
        print(f"  Kamera-Seitenverhältnis: {camera_aspect_ratio:.3f}")
        
        # Fester Rand von der Kamerakante (in normalisierten Koordinaten)
        fixed_margin = 0.08  # 8% Rand auf allen Seiten
        
        # Zusätzlicher Rand für Monitorschaltzonen, wenn mehrere Monitore vorhanden sind
        extra_horizontal_margin = 0.0
        if len(self.monitors) > 1:
            extra_horizontal_margin = self.monitor_switch_zone_width  # Platz für Monitorschaltzonen reservieren
        
        # Verfügbarer Platz für die Steuerungsregion
        available_width = 1.0 - 2 * (fixed_margin + extra_horizontal_margin)
        available_height = 1.0 - 2 * fixed_margin
        
        print(f"  Fester Rand: {fixed_margin:.3f}")
        print(f"  Zusätzlicher horizontaler Rand für Schaltzonen: {extra_horizontal_margin:.3f}")
        print(f"  Verfügbarer Platz: {available_width:.3f} x {available_height:.3f}")
        
        # Berechne das größtmögliche Rechteck mit Monitor-Seitenverhältnis
        # das innerhalb des verfügbaren Platzes passt
        if monitor_aspect_ratio > (available_width / available_height):
            # Monitor ist relativ zum verfügbaren Platz breiter - Breite ist begrenzender Faktor
            region_width = available_width
            region_height = region_width / monitor_aspect_ratio
            print(f"  Breitenbegrenzt: volle verfügbare Breite verwenden")
        else:
            # Monitor ist relativ zum verfügbaren Platz höher - Höhe ist begrenzender Faktor
            region_height = available_height
            region_width = region_height * monitor_aspect_ratio
            print(f"  Höhenbegrenzt: volle verfügbare Höhe verwenden")
        
        print(f"  Berechnete Region: {region_width:.3f} x {region_height:.3f}")
        print(f"  Seitenverhältnis der Region: {region_width/region_height:.3f}")
        print(f"  Seitenverhältnis des Monitors: {monitor_aspect_ratio:.3f}")
        print(f"  Unterschied im Seitenverhältnis: {abs((region_width/region_height) - monitor_aspect_ratio):.6f}")
        
        # Zentriere die Region im Kamerabild
        horizontal_center = 0.5
        vertical_center = 0.5
        
        self.region_x_min = horizontal_center - region_width / 2
        self.region_x_max = horizontal_center + region_width / 2
        self.region_y_min = vertical_center - region_height / 2
        self.region_y_max = vertical_center + region_height / 2
        
        print(f"  Steuerungsbereichsgrenzen: X({self.region_x_min:.3f}-{self.region_x_max:.3f}), Y({self.region_y_min:.3f}-{self.region_y_max:.3f})")
        
        # Überprüfe, ob das Seitenverhältnis genau korrekt ist
        final_aspect = (self.region_x_max - self.region_x_min) / (self.region_y_max - self.region_y_min)
        print(f"  Endgültiges Seitenverhältnis: {final_aspect:.6f}")
        print(f"  Perfekte Übereinstimmung des Seitenverhältnisses: {abs(final_aspect - monitor_aspect_ratio) < 0.001}")
        
        # Überprüfe, ob die Ränder eingehalten werden
        margin_left = self.region_x_min
        margin_right = 1.0 - self.region_x_max
        margin_top = self.region_y_min
        margin_bottom = 1.0 - self.region_y_max
        print(f"  Tatsächliche Ränder: L:{margin_left:.3f}, R:{margin_right:.3f}, T:{margin_top:.3f}, B:{margin_bottom:.3f}")
        
        # Überprüfe, ob die Mindestabstände eingehalten werden
        min_margin_check = all([
            margin_left >= fixed_margin + extra_horizontal_margin - 0.001,
            margin_right >= fixed_margin + extra_horizontal_margin - 0.001,
            margin_top >= fixed_margin - 0.001,
            margin_bottom >= fixed_margin - 0.001
        ])
        print(f"  Mindestabstände eingehalten: {min_margin_check}")
    
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
                
                # Verwende Mittelhand-MCP (Gelenk 9) anstelle der Spitze für stabilere Steuerung
                middle_finger_control = hand_landmarks.landmark[9]  # Wie ursprünglich
                
                # Begrenze die Erkennung auf die definierte Region
                norm_x = (middle_finger_control.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
                norm_y = (middle_finger_control.y - self.region_y_min) / (self.region_y_max - self.region_y_min)
                
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                
                # Konvertiere in die aktuellen Monitor-Koordinaten
                x, y = self.convert_to_monitor_coordinates(norm_x, norm_y, self.current_monitor)
                
                # Position glätten
                raw_position = (x, y)
                self.mouse_position = self.smooth_position(raw_position)
                self.mouse_action = "move"                  # Gestenerkennung
                self.detect_monitor_zone_switch(hand_landmarks)
                self.detect_3d_click(hand_landmarks)
                
                # Zeichne Handmarkierungen - wie ursprünglich
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
              # Zeichne Steuerungsregion und 3D-Tiefeninformationen
            region_start_x = int(self.region_x_min * self.camera_width)
            region_end_x = int(self.region_x_max * self.camera_width)
            region_start_y = int(self.region_y_min * self.camera_height)
            region_end_y = int(self.region_y_max * self.camera_height)
            cv2.rectangle(frame, (region_start_x, region_start_y), 
                         (region_end_x, region_end_y), (0, 255, 0), 2)
              # Zeichne Monitorschaltzonen (linke und rechte Kanten, angepasst an die grüne Steuerungsregion)
            if len(self.monitors) > 1:
                # Berechne Zonenpositionen basierend auf den Grenzen der grünen Steuerungsregion
                gap_width = 0.1  # 10% Lücke zwischen grünem Steuerbereich und roten Wechselzonen
                zone_width = 0.08  # 8% Breite für jede Schaltzone
                
                # Linke Zone beginnt an der linken Kante, endet vor der grünen Steuerregion (mit Lücke)
                left_zone_start = 0
                left_zone_end = max(0, self.region_x_min - gap_width)
                left_zone_width = left_zone_end - left_zone_start
                
                # Rechte Zone beginnt nach der grünen Steuerregion (mit Lücke), endet an der rechten Kante
                right_zone_start = min(1.0, self.region_x_max + gap_width)
                right_zone_end = 1.0
                right_zone_width = right_zone_end - right_zone_start
                
                # Konvertiere in Pixelkoordinaten
                left_zone_start_px = int(left_zone_start * self.camera_width)
                left_zone_end_px = int(left_zone_end * self.camera_width)
                right_zone_start_px = int(right_zone_start * self.camera_width)
                right_zone_end_px = int(right_zone_end * self.camera_width)
                
                # Zeichne linke Zone (nur wenn sie eine angemessene Breite hat)
                if left_zone_width > 0.05:  # Mindestens 5% Breite
                    cv2.rectangle(frame, (left_zone_start_px, 0), (left_zone_end_px, self.camera_height), (0, 0, 255), 2)
                    cv2.putText(frame, "← LINKS", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Zeichne rechte Zone (nur wenn sie eine angemessene Breite hat)
                if right_zone_width > 0.05:  # Mindestens 5% Breite
                    cv2.rectangle(frame, (right_zone_start_px, 0), (right_zone_end_px, self.camera_height), (0, 0, 255), 2)
                    cv2.putText(frame, "RECHTS →", (self.camera_width - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)# Zeige 3D-Tiefen- und Quetschdistanzinformationen
            if hasattr(self, 'smoothed_z'):
                depth_text = f"Tiefe: {self.smoothed_z:.3f}"
                cv2.putText(frame, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Zeige aktuelle Monitorinformationen
                monitor_text = f"Monitor: {self.current_monitor + 1}/{len(self.monitors)}"
                cv2.putText(frame, monitor_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Visuelle Tiefenanzeige
                depth_bar_length = int(abs(self.smoothed_z) * self.depth_scale)
                depth_bar_length = min(200, max(0, depth_bar_length))
                color = (0, 255, 0) if self.smoothed_z < 0 else (0, 0, 255)
                cv2.rectangle(frame, (10, 80), (10 + depth_bar_length, 100), color, -1)
                  # Zeige 3D-Quetschdistanz, wenn die Hand erkannt wird
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    pinch_distance = self.calculate_3d_distance(thumb_tip, index_tip)
                    
                    pinch_text = f"3D Quetschen: {pinch_distance:.3f}"
                    cv2.putText(frame, pinch_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Visuelle Quetschanzeige
                    pinch_color = (0, 0, 255) if pinch_distance < self.pinch_threshold_3d else (255, 255, 255)
                    cv2.circle(frame, (10, 140), 10, pinch_color, -1)                    # Zeige Wischgestenstatus
                    middle_finger_control = hand_landmarks.landmark[9]
                    norm_x = (middle_finger_control.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
                    norm_x = max(0, min(1, norm_x))
                    
                    hand_pos_text = f"Steuerung X: {norm_x:.2f}"
                    cv2.putText(frame, hand_pos_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # Zeige statusbasierte Monitorschaltanzeige
                    # Verwende den GLEICHEN Punkt, der die Maus steuert (Mittelhand-MCP-Gelenk 9)
                    middle_finger_control = hand_landmarks.landmark[9]
                    hand_x = middle_finger_control.x
                      # Berechne adaptive Zonen-Grenzen (gleich wie im Zeichencode)
                    gap_width = 0.05
                    left_zone_end = max(0, self.region_x_min - gap_width)
                    right_zone_start = min(1.0, self.region_x_max + gap_width)
                    
                    if hand_x < left_zone_end and left_zone_end > 0.05:
                        cv2.putText(frame, "ZONE: LINKS ←", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    elif hand_x > right_zone_start and (1.0 - right_zone_start) > 0.05:
                        cv2.putText(frame, "ZONE: RECHTS →", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "STEUERUNGSBEREICH", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Einfacher Fenstertitel wie ursprünglich
            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracking = HandTracking()
    try:
        hand_tracking.run()
    except KeyboardInterrupt:
        print("Skript wurde beendet!")
        pass