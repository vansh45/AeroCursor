import cv2
import mediapipe as mp
import uinput
from pynput import keyboard

device = uinput.Device([
    uinput.BTN_LEFT,
    uinput.REL_X,
    uinput.REL_Y,
])

mp_hands = mp.solutions.hands
exit_program = False
current_keys = set()

def on_press_key(key):
    current_keys.add(key)
    global exit_program
    if key == keyboard.Key.esc and (keyboard.Key.ctrl_l in current_keys or keyboard.Key.ctrl_r in current_keys):
        exit_program = True

def on_release_key(key):
    if key in current_keys:
        current_keys.remove(key)

listener = keyboard.Listener(on_press=on_press_key, on_release=on_release_key)
listener.start()

def run_webcam(camera_index=0):
    global exit_program
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        screen_w = 1920
        screen_h = 1080
        prev_x = screen_w // 2
        prev_y = screen_h // 2
        holding = False

        while True:
            if exit_program:
                break

            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                px = int(lm[12].x * screen_w)
                py = int(lm[12].y * screen_h)
                dx = px - prev_x
                dy = py - prev_y
                prev_x = px
                prev_y = py
                device.emit(uinput.REL_X, dx, syn=False)
                device.emit(uinput.REL_Y, dy)

                cond = (
                    lm[4].y < lm[14].y and
                    lm[6].y < lm[15].y and
                    lm[7].y < lm[16].y and
                    lm[8].y < lm[17].y and
                    lm[10].y < lm[18].y and
                    lm[11].y < lm[19].y and
                    lm[12].y < lm[20].y
                )
            else:
                cond = False

            if cond and not holding:
                device.emit(uinput.BTN_LEFT, 1)
                holding = True
            elif not cond and holding:
                device.emit(uinput.BTN_LEFT, 0)
                holding = False

            cv2.imshow("Hand Control", out)
            key = cv2.waitKey(1) & 255
            if key == 27 or exit_program:
                break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    run_webcam(2)
