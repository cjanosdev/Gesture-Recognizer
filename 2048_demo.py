from enum import Enum
import threading
import time

import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult, RunningMode
from mediapipe.tasks.python import BaseOptions

import pyautogui

class Gestures(Enum):
    THUMB_UP = "Go Up (Thumb Up)"
    OPEN_PALM = "Go Down (Open Palm)"
    POINTING_UP = "Go Left (Pointing Up)"
    VICTORY = "Go Right (Victory)"
    CLOSED_FIST = "Separate Movement (Closed Fist)"
    THUMB_DOWN = "Start New Game (Thumb Down)"

# Path to the gesture recognition model
GESTURE_MODEL = "gesture_recognizer.task" 


# Note from (https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#live-stream_1)
# If you use the live stream mode, you’ll need to register a result listener
# when creating the task. The listener is called whenever the task 
# has finished processing a video frame with the detection result 
# and the input image as parameters.
def livestream_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Defaults
    gesture_text = "Unrecognized Gesture"
    raw_gesture = None
    confidence = 0.0
    hand_landmarks = result.hand_landmarks if result.hand_landmarks else []


    if result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
        raw_gesture = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        gesture_text = friendly_name(raw_gesture)
    
    # Call the function to handle the gesture action based on the recognized gesture
    handle_gesture_action(raw_gesture)

    # Update the latest result with the new gesture info
    with result_lock:
        latest_result["hand_landmarks"] = hand_landmarks
        latest_result["gesture_text"] = gesture_text
        latest_result["raw_gesture"] = raw_gesture
        latest_result["confidence"] = confidence
        latest_result["timestamp_ms"] = timestamp_ms

# Initialize the Gesture Recognizer
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#run_the_task
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=livestream_callback
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Result state from livestream callback stuff
latest_result = {
    "hand_landmarks": [],
    "gesture_text": "Unrecognized Gesture",
    "raw_gesture": None,
    "confidence": 0.0,
    "timestamp_ms": 0,
}

# Lock for synchronizing access to the latest_result
result_lock = threading.Lock()

# Gesture trigger cooldowns
# Helps with live stream not being super jittery/janky
last_action_time = 0.0
ACTION_COOLDOWN = 0.75
CLOSED_FIST_COOLDOWN = 1.0
RESTART_GAME_COOLDOWN = 2.0

# Helper funciton to get friendly gesture name for displaying
def friendly_name(recognized_gesture: str) -> str:
    return {
        "Thumb_Up": Gestures.THUMB_UP.value,
        "Open_Palm": Gestures.OPEN_PALM.value,
        "Pointing_Up": Gestures.POINTING_UP.value,
        "Victory": Gestures.VICTORY.value,
        "Closed_Fist": Gestures.CLOSED_FIST.value,
        "Thumb_Down": Gestures.THUMB_DOWN.value,
    }.get(recognized_gesture, "Unrecognized Gesture")

# Helper function to check if we can trigger a new action based on cooldown
def can_trigger(cooldown: float) -> bool:
    # Uses global variable to track last action time 
    # and checks if enough time has passed based on the
    #  provided cooldown
    global last_action_time
    now = time.time()
    if now - last_action_time >= cooldown:
        last_action_time = now
        return True
    return False

def handle_gesture_action(recognized_gesture: str) -> None:
    # This function takes in the recognized gesture and performs the corresponding action
    # It also checks for cooldowns to prevent spamming actions and improve user experience
    if recognized_gesture == "Thumb_Up" and can_trigger(ACTION_COOLDOWN):
        pyautogui.press("up")

    elif recognized_gesture == "Open_Palm" and can_trigger(ACTION_COOLDOWN):
        pyautogui.press("down")

    elif recognized_gesture == "Pointing_Up" and can_trigger(ACTION_COOLDOWN):
        pyautogui.press("left")

    elif recognized_gesture == "Victory" and can_trigger(ACTION_COOLDOWN):
        pyautogui.press("right")

    elif recognized_gesture == "Closed_Fist" and can_trigger(CLOSED_FIST_COOLDOWN):
        # Closed fist is used as a resting position, 
        # so we just want to trigger a cooldown here to prevent 
        # accidental triggers when the user is resting their hand
        pass

    elif recognized_gesture == "Thumb_Down" and can_trigger(RESTART_GAME_COOLDOWN):
        
        # These coordinates are based on full screen play on my mac m1 pro, 
        # So you may need to adjust these coordinates based on your screen resolution 
        # and where the game is located on your screen
        # tried using locateCenterOnScreen to find the buttons but it was too slow, 
        # and also not super accurate so hardcoding the coordinates for better performance
        # in this case we should demo from my laptop

        # New game button location from my screen.
        new_game_btn = (1091, 175)

        # Move the mouse to the new game button and click it
        pyautogui.moveTo(new_game_btn[0], new_game_btn[1], duration=0.2)
        time.sleep(0.2)
        pyautogui.click(new_game_btn[0], new_game_btn[1])
        time.sleep(0.2)

        # Start game button location from my screen.
        start_game_btn = (758, 590)

        # Move the mouse to the start game button and click it
        pyautogui.moveTo(start_game_btn[0], start_game_btn[1], duration=0.2)
        time.sleep(0.2)
        pyautogui.click(start_game_btn[0], start_game_btn[1])


# Draws hand landmarks and connections on the image 
# using the Mediapipe Tasks API connection objects
def draw_hand_landmarks(image, hand_landmarks):
    h,w, _ = image.shape

    # all hand connections : https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarksConnections#class-variables
    # [HandLandmarksConnections.Connection(start=0, end=1),
    # HandLandmarksConnections.Connection(start=1, end=5),
    # HandLandmarksConnections.Connection(start=9, end=13),
    # HandLandmarksConnections.Connection(start=13, end=17),
    # HandLandmarksConnections.Connection(start=5, end=9),
    # HandLandmarksConnections.Connection(start=0, end=17),
    # HandLandmarksConnections.Connection(start=1, end=2),
    # HandLandmarksConnections.Connection(start=2, end=3),
    # HandLandmarksConnections.Connection(start=3, end=4),
    # HandLandmarksConnections.Connection(start=5, end=6),
    # HandLandmarksConnections.Connection(start=6, end=7),
    # HandLandmarksConnections.Connection(start=7, end=8),
    # HandLandmarksConnections.Connection(start=9, end=10),
    # HandLandmarksConnections.Connection(start=10, end=11),
    # HandLandmarksConnections.Connection(start=11, end=12),
    # HandLandmarksConnections.Connection(start=13, end=14),
    # HandLandmarksConnections.Connection(start=14, end=15),
    # HandLandmarksConnections.Connection(start=15, end=16),
    # HandLandmarksConnections.Connection(start=17, end=18),
    # HandLandmarksConnections.Connection(start=18, end=19),
    # HandLandmarksConnections.Connection(start=19, end=20)]
    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        start_idx = connection.start
        end_idx = connection.end

        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]

        #  Normalized Landmark represents a point in 3D space with x, y, z coordinates.
        #  x and y are normalized to [0.0, 1.0] by the image width and height respectively.
        #  z represents the landmark depth, and the smaller the value the closer 
        #  the landmark is to the camera. 
        #  The magnitude of z uses roughly the same scale as x.
        x0, y0 = int(start.x * w), int(start.y * h)
        x1, y1 = int(end.x * w), int(end.y * h)

        # Draws connections as green lines
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 5, cv2.LINE_AA)

    # Draws landmarks as red circles
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)




def main():
     # Initialize video capture
    cap = cv2.VideoCapture(0) 

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # When running in the video mode or the live stream mode, 
        # I need to provide the Gesture Recognizer task the 
        # timestamp of the input frame so thats why I have this here
        timestamp_ms = int(time.time() * 1000)
        
        # Have to use recognize_async for live stream mode per docs and it runs better in livestream mode on my computer
        #  https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#run_the_task
        # # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        gesture_recognizer.recognize_async(mp_image, timestamp_ms)

        with result_lock:
                hand_landmarks_list = latest_result["hand_landmarks"]
                gesture_text = latest_result["gesture_text"]
                confidence = latest_result["confidence"]

        # Draw landmarks
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                draw_hand_landmarks(image, hand_landmarks)
        
         # Display recognized gesture and confidence 
        cv2.putText(image, f"Gesture: {gesture_text} ({confidence:.2f})", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        # Display the resulting image (can comment this out for better performance later on)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()