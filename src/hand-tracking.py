import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import os

class HandTrackingMouseControl:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1, static_image_mode=False)
        
        # Works in Windows & Linux/Mac
        os.system("cls" if os.name == "nt" else "clear")
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
        self.mouse_position = (0, 0)
        self.mouse_action = None
        self.sensi_down = 0.045  # Sensitivity for click detection
        self.sensi_up = 0.05
        
        # Disable PyAutoGUI Fail-Safe (use at your own risk!)
        pyautogui.FAILSAFE = False
        
        # Start the separate thread for mouse control
        self.mouse_thread = threading.Thread(target=self.handle_mouse, daemon=True)
        self.mouse_thread.start()
    
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
                        # self.click_start_time = time.time()
                        pyautogui.mouseDown()
                        self.holding = True
                elif distance > self.sensi_up:
                    if self.holding:
                        pyautogui.mouseUp()
                        self.holding = False
                    self.clicking = False  # Reset click state
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracking = HandTrackingMouseControl()
    hand_tracking.run()
