import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import os
import sys
from pathlib import Path

# ASL-Module als globale Variablen
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
        print("press strg or ctrl + c to exit")
        
        # ASL Setup
        self.enable_asl = enable_asl and ASL_AVAILABLE
        self.asl_model = None
        if self.enable_asl:
            self._setup_asl()
        else:
            self._setup_asl()
        
        # Modi-Wechsel Setup
        self.asl_mode = False  # False = Maus, True = ASL-Tastatureingaben
        self.mode_switch_cooldown = 0
        self.mode_switch_delay = 1.5  # 1.5 Sekunden Cooldown
        self.mode_switching = False  # Status für Modi-Wechsel
        
        # ASL-Tastatureingaben Setup
        self.last_typed_letter = ""
        self.letter_cooldown = 0
        self.letter_delay = 1.0  # 1 Sekunde zwischen Buchstaben

        self.cap = cv2.VideoCapture(0)
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.camera_width, self.camera_height = 640, 480  # Standard camera resolution, adjust if necessary
        
        # Define the region of the camera used for control
        self.region_x_min, self.region_x_max = 0.2, 0.8  # Use only the middle part of the camera
        self.region_y_min, self.region_y_max = 0.2, 0.8
        
        self.clicking = False  # State for mouse button pressed
        self.click_start_time = None
        self.holding = False  # State for mouse button held
        self.mouse_position = (0, 0)
        self.mouse_action = None
        self.sensi_down = 0.045  # Sensitivity for click detection
        self.sensi_up = 0.05
        
        # Disable PyAutoGUI Fail-Safe (use at your own risk!)
        pyautogui.FAILSAFE = False
        
        # Start the separate thread for mouse control
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
    
    def _setup_asl(self):
        """Setup für ASL-Erkennung - GENAU WIE TEST_CAMERA"""
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
            print(f"Fehler beim ASL-Import: {e}")
            self.enable_asl = False
            return
            
        model_path = Path(__file__).parent.parent / "asl" / "model" / "frame_mlp_asl.pt"
        
        if not model_path.exists():
            print(f"ASL-Modell nicht gefunden: {model_path}")
            self.enable_asl = False
            return
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            asl_model = FrameMLP2()
            asl_model.load_state_dict(torch.load(model_path, map_location=device))
            asl_model.to(device)
            asl_model.eval()
            self.asl_model = asl_model
            
            # Nur 26 Buchstaben (A-Z) wie im Modell trainiert
            self.class_names = [chr(ord('A') + i) for i in range(26)]
            
            self.prediction_history = []
            self.current_prediction = ""
            self.prediction_confidence = 0.0
            
            print(f"ASL-Modell geladen! (Device: {self.device})")
        except Exception as e:
            print(f"Fehler beim Laden des ASL-Modells: {e}")
            self.enable_asl = False
    
    def _extract_landmarks(self, hand_landmarks):
        """Extrahiert Hand-Landmarks für ASL"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def _predict_asl(self, landmarks):
        """EXAKT WIE TEST_CAMERA mit Debug"""
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.asl_model(x)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
        return self.class_names[pred_idx], confidence
    
    def _smooth_asl_prediction(self, letter, confidence):
        """EXAKT wie in test_camera.py"""
        self.prediction_history.append((letter, confidence))
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Häufigster Buchstabe mit hoher Confidence
        letter_counts = {}
        total_confidence = 0
        
        for l, c in self.prediction_history[-5:]:  # Nur letzte 5 Frames
            if c > 0.7:  # Nur hohe Confidence
                letter_counts[l] = letter_counts.get(l, 0) + 1
                total_confidence += c
        
        if letter_counts:
            most_common = max(letter_counts.items(), key=lambda x: x[1])
            avg_conf = total_confidence / len(self.prediction_history[-5:])
            return most_common[0], avg_conf
        
        return letter, confidence
    



    
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
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Only process the first detected hand
                
                # Modus-Wechsel Geste (EXAKT wie Daumen+Zeigefinger Klick!)
                thumb_tip = hand_landmarks.landmark[4]  # Daumen-Spitze
                middle_finger = hand_landmarks.landmark[12]  # Mittelfinger-Spitze
                distance = abs(thumb_tip.x - middle_finger.x) + abs(thumb_tip.y - middle_finger.y)
                
                if distance <= self.sensi_down:  # Wenn Daumen und Mittelfinger sich berühren
                    if not self.mode_switching:
                        self.mode_switching = True
                        self.asl_mode = not self.asl_mode
                        mode_name = "ASL-Tastatur" if self.asl_mode else "Maussteuerung"
                        print(f"Modus gewechselt zu: {mode_name}")
                elif distance > self.sensi_up:
                    if self.mode_switching:
                        self.mode_switching = False
                
                # ASL-Erkennung (nur im ASL-Modus)
                if self.enable_asl and self.asl_mode:
                    # Landmarks extrahieren und vorhersagen
                    landmarks = self._extract_landmarks(hand_landmarks)
                    letter, confidence = self._predict_asl(landmarks)
                    
                    # Prediction glätten
                    smooth_letter, smooth_confidence = self._smooth_asl_prediction(letter, confidence)
                    
                    self.current_prediction = smooth_letter
                    self.prediction_confidence = smooth_confidence
                    
                    # Tippen bei hoher Confidence
                    if smooth_confidence > 0.8:
                        if not hasattr(self, 'last_letter_time'):
                            self.last_letter_time = 0
                        if not hasattr(self, 'last_letter'):
                            self.last_letter = ""
                            
                        current_time = time.time()
                        if current_time - self.last_letter_time > 1.0 and smooth_letter != self.last_letter:
                            pyautogui.write(smooth_letter.lower())
                            print(f"Getippt: {smooth_letter}")
                            self.last_letter_time = current_time
                            self.last_letter = smooth_letter
                
                # Maussteuerung (nur im Maus-Modus)
                if not self.asl_mode:
                    middle_finger_tip = hand_landmarks.landmark[9]  # Middle finger tip for mouse control
                    
                    # Limit the detection area to the defined part of the camera
                    norm_x = (middle_finger_tip.x - self.region_x_min) / (self.region_x_max - self.region_x_min)
                    norm_y = (middle_finger_tip.y - self.region_y_min) / (self.region_y_max - self.region_y_min)
                    
                    # Ensure values are within valid bounds
                    norm_x = max(0, min(1, norm_x))
                    norm_y = max(0, min(1, norm_y))
                    
                    x = int(norm_x * self.screen_width)
                    y = int(norm_y * self.screen_height)
                    
                    self.mouse_position = (x, y)
                    self.mouse_action = "move"
                    
                    index_finger = hand_landmarks.landmark[8]  # Index finger tip
                    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                    distance = abs(index_finger.x - thumb_tip.x) + abs(index_finger.y - thumb_tip.y)
                    
                    if distance <= self.sensi_down:  # If thumb and index finger touch
                        if not self.clicking:
                            self.clicking = True
                            pyautogui.mouseDown()
                            self.holding = True
                    elif distance > self.sensi_up:
                        if self.holding:
                            pyautogui.mouseUp()
                            self.holding = False
                        self.clicking = False  # Reset click state
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Modus und Steuerung anzeigen
            mode_text = "ASL-Tastatur-Modus" if self.asl_mode else "Maus-Modus"
            mode_color = (0, 255, 0) if self.asl_mode else (0, 255, 255)
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            cv2.putText(frame, "Daumen+Mittelfinger = Modus wechseln", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # ASL-Infos anzeigen (nur im ASL-Modus)
            if self.enable_asl and self.asl_mode:
                y_offset = 90
                if hasattr(self, 'current_prediction'):
                    confidence_text = f"({self.prediction_confidence:.2f})" if hasattr(self, 'prediction_confidence') else ""
                    cv2.putText(frame, f"Buchstabe: {self.current_prediction} {confidence_text}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Typing Buffer anzeigen
                if hasattr(self, 'typing_buffer') and self.typing_buffer:
                    cv2.putText(frame, f"Buffer: {self.typing_buffer}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                # Gesten-Hilfe
                cv2.putText(frame, "Daumen+Kleinfinger = Leerzeichen", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, "Zeigef.+Ringf. = Backspace", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, "Alle Finger zusammen = Enter", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Confidence: {self.prediction_confidence:.2f}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Hinweise für Steuerung
            cv2.putText(frame, "Daumen+Mittelfinger = Modus | M = Modus | S = Space", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Kontrollbereich zeichnen (nur im Maus-Modus)
            if not self.asl_mode:
                height, width = frame.shape[:2]
                x1 = int(self.region_x_min * width)
                y1 = int(self.region_y_min * height)
                x2 = int(self.region_x_max * width)
                y2 = int(self.region_y_max * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Mausbereich", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Hand Tracking", frame)
            
            # Tastatur-Eingaben verarbeiten
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):  # 'M'-Taste für Modus-Wechsel
                self.asl_mode = not self.asl_mode
                mode_name = "ASL-Tastatur" if self.asl_mode else "Maussteuerung"
                print(f"Modus gewechselt zu: {mode_name}")
            elif key == ord('s') and self.asl_mode:  # 'S'-Taste für Leerzeichen
                pyautogui.write(' ')
                print("Leerzeichen getippt")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand Tracking mit ASL-Erkennung")
    parser.add_argument("--no-asl", action="store_true", help="ASL-Erkennung deaktivieren")
    args = parser.parse_args()
    
    hand_tracking = HandTrackingMouseControl(enable_asl=not args.no_asl)
    try:
        hand_tracking.run()
    except KeyboardInterrupt:
        print("Script has been terminated!")
        pass
