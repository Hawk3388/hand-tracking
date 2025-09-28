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
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1, static_image_mode=False)
        
        # ASL Setup
        self.enable_asl = enable_asl and ASL_AVAILABLE
        self.asl_model = None
        if self.enable_asl:
            self._setup_asl()
        else:
            self._setup_asl()
        
        # Modi-Wechsel Setup
        self.asl_mode = False  # False = Maus, True = ASL
        self.mode_switch_cooldown = 0
        self.mode_switch_delay = 1.5  # 1.5 Sekunden Cooldown

        # Works in Windows & Linux/Mac
        time.sleep(0.001)
        os.system("cls" if os.name == "nt" else "clear")
        print("Hand Tracking (Background)" + (" + ASL" if self.enable_asl else ""))
        print("press strg or ctrl + c to exit")
        
        self.cap = cv2.VideoCapture(0)
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.camera_width, self.camera_height = 640, 480  # Standard camera resolution, adjust if necessary
        
        # Define the region of the camera used for control
        self.region_x_min, self.region_x_max = 0.2, 0.8  # Use only the middle part of the camera
        self.region_y_min, self.region_y_max = 0.2, 0.8
        
        self.clicking = False  # State for mouse button pressed
        self.click_start_time = None
        self.holding = False  # State for mouse button held
        self.hand_recognize = False
        self.mouse_position = (0, 0)
        self.start = True
        self.mouse_action = None
        self.sensi_down = 0.045  # Sensitivity for click detection
        self.sensi_up = 0.05
        
        # Disable PyAutoGUI Fail-Safe (use at your own risk!)
        pyautogui.FAILSAFE = False
        
        # Start the separate thread for mouse control
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
    
    def _setup_asl(self):
        """Setup für ASL-Erkennung"""
        # ASL Import - genau wie main version
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
            self.last_prediction_time = 0
            
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
        """Vorhersage für ASL-Buchstaben"""
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.asl_model(x)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        return self.class_names[pred_idx], confidence
    
    def _smooth_asl_prediction(self, letter, confidence):
        """Glättet ASL-Vorhersagen"""
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
    
    def _detect_mode_switch_gesture(self, hand_landmarks):
        """Erkennt Daumen + kleiner Finger Geste für Modus-Wechsel"""
        if time.time() < self.mode_switch_cooldown:
            return False
        
        # Daumen-Spitze (4) und kleiner Finger-Spitze (20)
        thumb_tip = hand_landmarks.landmark[4]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Distanz zwischen Daumen und kleinem Finger
        distance = abs(thumb_tip.x - pinky_tip.x) + abs(thumb_tip.y - pinky_tip.y)
        
        # Wenn sie sich berühren (sehr nah sind)
        if distance < 0.05:  # Empfindlichkeit für Gestenerkennung
            self.mode_switch_cooldown = time.time() + self.mode_switch_delay
            return True
        
        return False
    
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

            if self.start:
                self.start = False
                print("Hand control activated!")
        
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)  # Flip for natural control
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))  # Ensure size matches
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                if not self.hand_recognize:
                    self.hand_recognize = True
                    print("Hand detected")

                hand_landmarks = results.multi_hand_landmarks[0]  # Only process the first detected hand
                
                # Modus-Wechsel Geste erkennen
                if self.enable_asl and self._detect_mode_switch_gesture(hand_landmarks):
                    self.asl_mode = not self.asl_mode
                    mode_name = "ASL-Erkennung" if self.asl_mode else "Maussteuerung"
                    print(f"Modus gewechselt zu: {mode_name}")
                
                # ASL-Erkennung (nur im ASL-Modus und alle 0.5 Sekunden ausgeben)
                if self.enable_asl and self.asl_mode and time.time() - self.last_prediction_time > 0.5:
                    landmarks = self._extract_landmarks(hand_landmarks)
                    letter, confidence = self._predict_asl(landmarks)
                    if confidence > 0.8:  # Nur bei hoher Confidence ausgeben
                        print(f"ASL erkannt: {letter} (Confidence: {confidence:.2f})")
                        self.last_prediction_time = time.time()
                
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
            else:
                if self.hand_recognize:
                    self.hand_recognize = False
                    print("No hand detected")
                
                # Maus loslassen wenn keine Hand erkannt
                if self.holding:
                    pyautogui.mouseUp()
                    self.holding = False
                    self.clicking = False
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand Tracking (Background) mit ASL-Erkennung")
    parser.add_argument("--no-asl", action="store_true", help="ASL-Erkennung deaktivieren")
    args = parser.parse_args()
    
    hand_tracking = HandTrackingMouseControl(enable_asl=not args.no_asl)
    try:
        hand_tracking.run()
    except KeyboardInterrupt:
        print("Script has been terminated!")
        pass
