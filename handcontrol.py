import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

# ---------------- Screen Setup ----------------
screen_w, screen_h = pyautogui.size()
# Disable PyAutoGUI fail-safe to prevent corner detection issues
pyautogui.FAILSAFE = False

# ---------------- Hand Detection Setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ---------------- Audio Setup ----------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(0)

# Cursor smoothing
prev_x, prev_y = 0, 0
smoothening = 8  # higher = smoother movement
cursor_speed = 3  # scale movement speed (reduced to prevent getting stuck)
volume_locked = False
click_cooldown = 0.3  # seconds between clicks
last_click_time = 0
last_double_click_time = 0
double_click_threshold = 0.5  # seconds for double click
last_right_click_time = 0
scroll_cooldown = 0.1  # seconds between scroll actions
last_scroll_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        # Get hand landmarks and hand type (left/right)
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Get hand type (left or right)
            hand_type = result.multi_handedness[idx].classification[0].label
            
            # Draw landmarks with different colors for left/right hands
            if hand_type == "Left":
                color = (0, 255, 0)  # Green for left hand
                hand_text = "LEFT"
            else:
                color = (255, 0, 0)  # Red for right hand
                hand_text = "RIGHT"
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=color, thickness=2))
            
            # Display hand type on screen
            cv2.putText(frame, f"{hand_text} HAND", (50, 150 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display gesture instructions
        cv2.putText(frame, "Gestures:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, "Middle finger = Click", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "Ring finger = Right Click", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "Thumb up/down = Scroll", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "Pinkie up = Volume Lock", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        h, w, _ = frame.shape
        # ---------------- Cursor Control (using first detected hand) ----------------
        first_hand = result.multi_hand_landmarks[0]
        index_tip = first_hand.landmark[8]  # Index finger tip
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Create centered control box (smaller box in center of camera)
        box_w = w // 3  # One third camera width
        box_h = h // 3  # One third camera height
        box_x = w // 3  # Start x position (centered)
        box_y = h // 3  # Start y position (centered)
        
        # Draw the control box on camera view
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
        
        # Map finger position within the control box to full screen
        if box_x <= x <= box_x + box_w and box_y <= y <= box_y + box_h:
            # Finger is within the control box
            screen_x = np.interp(x, [box_x, box_x + box_w], [0, screen_w])
            screen_y = np.interp(y, [box_y, box_y + box_h], [0, screen_h])
        else:
            # Finger is outside control box - don't move cursor
            screen_x = prev_x
            screen_y = prev_y
        # Increase cursor speed by multiplying difference
        curr_x = prev_x + (screen_x - prev_x) * cursor_speed / smoothening
        curr_y = prev_y + (screen_y - prev_y) * cursor_speed / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y
        
        # Enhanced visual feedback
        cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)  # Larger cursor
        cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)   # Outer ring
        cv2.circle(frame, (x, y), 25, (255, 255, 0), 1) # Glow effect

        # ---------------- Volume Control ----------------
        thumb_tip = first_hand.landmark[4]
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = x, y  # index tip coordinates
        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.circle(frame, (x1, y1), 10, (0,0,255), -1)

        # Pinkie-based lock/unlock
        pinkie_tip = first_hand.landmark[20]
        pinkie_pip = first_hand.landmark[18]
        if pinkie_tip.y < pinkie_pip.y:  # pinkie up = lock
            volume_locked = True
            cv2.putText(frame, "Volume Locked", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:  # pinkie down = unlock
            volume_locked = False

        if not volume_locked:
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            min_dist, max_dist = 20, 200
            vol = np.interp(distance, [min_dist, max_dist], [min_vol + (max_vol - min_vol)*0.3, max_vol])
            volume.SetMasterVolumeLevel(vol, None)
            vol_percent = np.interp(distance, [min_dist, max_dist], [30, 100])
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else:
            vol_db = volume.GetMasterVolumeLevel()
            vol_percent = np.interp(vol_db, [min_vol, max_vol], [0,100])
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # ---------------- Click Controls ----------------
        middle_tip = first_hand.landmark[12]
        middle_pip = first_hand.landmark[10]
        ring_tip = first_hand.landmark[16]
        ring_pip = first_hand.landmark[14]
        curr_time = time.time()
        
        # Left Click - Middle finger full bend
        if middle_tip.y > middle_pip.y and (curr_time - last_click_time) > click_cooldown:
            # Check for double click
            if (curr_time - last_double_click_time) < double_click_threshold:
                pyautogui.doubleClick()
                cv2.putText(frame, "Double Click!", (screen_w//2,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                last_double_click_time = 0  # Reset
            else:
                pyautogui.click()
                cv2.putText(frame, "Click!", (screen_w//2,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                last_double_click_time = curr_time
            last_click_time = curr_time
        
        # Right Click - Ring finger full bend
        if ring_tip.y > ring_pip.y and (curr_time - last_right_click_time) > click_cooldown:
            pyautogui.rightClick()
            last_right_click_time = curr_time
            cv2.putText(frame, "Right Click!", (screen_w//2,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
        # ---------------- Scroll Control ----------------
        # Use thumb vertical movement for scrolling
        thumb_y = thumb_tip.y
        if (curr_time - last_scroll_time) > scroll_cooldown:
            if thumb_y < 0.3:  # Thumb high = scroll up
                pyautogui.scroll(3)
                cv2.putText(frame, "Scroll Up", (screen_w//2,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                last_scroll_time = curr_time
            elif thumb_y > 0.7:  # Thumb low = scroll down
                pyautogui.scroll(-3)
                cv2.putText(frame, "Scroll Down", (screen_w//2,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                last_scroll_time = curr_time

    cv2.imshow("Hand Mouse + Volume Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif cv2.getWindowProperty("Hand Mouse + Volume Control", cv2.WND_PROP_VISIBLE) < 1:  # X button clicked
        break

cap.release()
cv2.destroyAllWindows()
