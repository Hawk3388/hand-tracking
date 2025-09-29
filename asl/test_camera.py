import torch
import cv2
import mediapipe as mp
import numpy as np
from model import FrameMLP2
import json

class ASLTester:
    def __init__(self, model_path="./asl/model/frame_mlp_asl.pt"):
        # Device setup
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Modell laden
        self.model = FrameMLP2()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Modell erfolgreich geladen!")
        except FileNotFoundError:
            print(f"Modell nicht gefunden: {model_path}")
            exit(1)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Klassennamen A-Z
        self.class_names = [chr(ord('A') + i) for i in range(26)]
        
        # Für Prediction-Glättung
        self.prediction_history = []
        self.history_size = 10
        
    def extract_landmarks(self, hand_landmarks):
        """Extrahiert Hand-Landmarks als flachen Array"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def predict_letter(self, landmarks):
        """Vorhersage für einen Satz von Landmarks"""
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
        return self.class_names[pred_idx], confidence
    
    def smooth_prediction(self, letter, confidence):
        """Glättet Vorhersagen über mehrere Frames"""
        self.prediction_history.append((letter, confidence))
        
        if len(self.prediction_history) > self.history_size:
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
            return most_common[0], total_confidence / len(self.prediction_history[-5:])
        
        return letter, confidence
    
    def run_camera_test(self):
        """Hauptschleife für Kamera-Test"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Fehler: Kamera konnte nicht geöffnet werden")
            return
        
        print("Kamera gestartet! Drücke 'q' zum Beenden, 's' zum Speichern eines Samples")
        print("Zeige deine Hand vor die Kamera...")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame spiegeln für bessere UX
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand-Detection
            results = self.hands.process(frame_rgb)
            
            prediction_text = "Keine Hand erkannt"
            confidence_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Landmarks zeichnen
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Landmarks extrahieren und vorhersagen
                    landmarks = self.extract_landmarks(hand_landmarks)
                    letter, confidence = self.predict_letter(landmarks)
                    
                    # Prediction glätten
                    smooth_letter, smooth_confidence = self.smooth_prediction(letter, confidence)
                    
                    prediction_text = f"Buchstabe: {smooth_letter}"
                    confidence_text = f"Confidence: {smooth_confidence:.3f}"
                    
                    # Speichern bei 's' Taste (für Debugging)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        self.save_sample(landmarks, letter, frame_count)
                        frame_count += 1
            
            # Text auf Frame zeichnen
            cv2.putText(frame, prediction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Druecke 'q' zum Beenden, 's' zum Speichern", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('ASL Hand Recognition Test', frame)
            
            # Beenden bei 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera-Test beendet.")
    
    def save_sample(self, landmarks, predicted_letter, frame_count):
        """Speichert ein Sample für Debugging"""
        # Landmarks als Liste von Dicts formatieren (wie im Training)
        landmarks_dict = []
        for i in range(0, len(landmarks), 3):
            landmarks_dict.append({
                'x': landmarks[i],
                'y': landmarks[i+1], 
                'z': landmarks[i+2]
            })
        
        filename = f"test_sample_{frame_count}_{predicted_letter}.json"
        with open(filename, 'w') as f:
            json.dump(landmarks_dict, f)
        print(f"Sample gespeichert: {filename}")

def main():
    print("ASL Hand Recognition Tester")
    print("=" * 30)
    
    # Tester initialisieren
    tester = ASLTester()
    
    # Kamera-Test starten
    tester.run_camera_test()

if __name__ == "__main__":
    main()
