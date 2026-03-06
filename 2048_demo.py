import time

import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions

import pyautogui

# Path to the gesture recognition model
GESTURE_MODEL = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
     # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

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

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)


        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            if recognized_gesture == "Thumb_Up":
                pyautogui.press("up")

                # Short delay to not spam the up gesture too many times
                # making the game unenjoyable to play
                time.sleep(0.75)  

            elif recognized_gesture == "Open_Palm":
                pyautogui.press("down")

                # Short delay to not spam the down gesture too many times
                # making the game unenjoyable to play
                time.sleep(0.75)

            elif recognized_gesture == "Pointing_Up":
                pyautogui.press("left")

                # Short delay to not spam the left gesture too many times
                # making the game unenjoyable to play
                time.sleep(0.75)

            elif recognized_gesture == "Victory":
                pyautogui.press("right")

                # Short delay to not spam the right gesture too many times
                # making the game unenjoyable to play
                time.sleep(0.75)

            elif recognized_gesture == "Closed_Fist":
                    # Long delay to just act as a
                    # an artifical delay between gestures
                    # to improve user experience
                    # we found fist was the most natural resting position
                    time.sleep(1.0)

            elif recognized_gesture == "Thumb_Down":
                    # These coordinates are based on full screen play on
                    # my mac m1 pro, you may need to adjust these coordinates based on your screen resolution and where the game is located on your screen
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
                    time.sleep(0.75)
        
            # Display recognized gesture and confidence 
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image (can comment this out for better performance later on)
        #cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()