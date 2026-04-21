#!/usr/bin/env python3
# coding=utf8
"""
Combined Avoidance — HiWonder TurboPi
Merges color_detection.py (traffic light logic) with the HiwonderSDK-based
obstacle avoidance script into one program.

SDK:  HiwonderSDK.mecanum / HiwonderSDK.Sonar / HiwonderSDK.ros_robot_controller_sdk
      (same packages used in the standalone avoidance script)

Thread architecture:
  VisionThread  — camera loop; writes detected_color to shared state
  Main loop     — reads sonar + detected_color; owns all motor commands

Priority (highest first):
  1. Sonar obstacle    → avoid_obstacle() maneuver (stop / reverse / turn)
  2. Traffic RED       → stop_car()
  3. Traffic YELLOW    → drive_forward at half speed
  4. Traffic GREEN     → drive_forward at full speed
  5. No signal (none)  → drive_forward at full speed
"""

import sys
sys.path.append('/home/pi/TurboPi/')

import cv2
import numpy as np
import threading
import time
import signal

import HiwonderSDK.Sonar                  as Sonar
import HiwonderSDK.mecanum                as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this script with python3!')
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Hardware objects  (same as obstacle avoidance script)
# ─────────────────────────────────────────────────────────────────────────────
car   = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
board = rrc.Board()

# ─────────────────────────────────────────────────────────────────────────────
# Avoidance / sonar parameters  (copied directly from obstacle avoidance script)
# ─────────────────────────────────────────────────────────────────────────────
FORWARD_SPEED         = 35
REVERSE_SPEED         = 30
SAFE_DISTANCE_CM      = 24.0
DIST_WINDOW           = 5
LOOP_DELAY            = 0.05

STOP_PAUSE            = 0.15
REVERSE_TIME          = 0.40
TURN_TIME             = 0.65
FORWARD_RECOVERY_TIME = 0.45
COOLDOWN_TIME         = 1.00

TURN_DIRECTION        = 'left'
TURN_YAW              = 0.45
DEBUG_PRINT           = True

# ─────────────────────────────────────────────────────────────────────────────
# Camera parameters  (copied from color_detection.py)
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
CAMERA_FPS        = 30

# ─────────────────────────────────────────────────────────────────────────────
# Color detection parameters  (copied from color_detection.py)
# ─────────────────────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA  = 1500
DETECTION_CONFIRM = 3
COLOR_DEBOUNCE_S  = 0.4

COLOR_RANGES = {
    "red": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),
    ],
    "yellow": [
        (np.array([18,  100,  80]),  np.array([35,  255, 255])),
    ],
    "green": [
        (np.array([36,   80,  60]),  np.array([89,  255, 255])),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared state between VisionThread and main loop
# ─────────────────────────────────────────────────────────────────────────────
detected_color = "none"      # written by VisionThread, read by main loop
shared_frame   = None        # annotated frame for display
color_lock     = threading.Lock()
frame_lock     = threading.Lock()
stop_event     = threading.Event()

last_avoid_time   = 0.0
last_distance_cm  = 999.0
running           = True


# ─────────────────────────────────────────────────────────────────────────────
# Motor helpers  (from obstacle avoidance script, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def stop_car():
    car.set_velocity(0, 90, 0)

def drive_forward(speed=FORWARD_SPEED):
    car.set_velocity(speed, 90, 0)

def drive_reverse(speed=REVERSE_SPEED):
    car.set_velocity(speed, 270, 0)

def turn_in_place(direction='left', yaw=TURN_YAW):
    if direction == 'left':
        car.set_velocity(0, 90, -abs(yaw))
    else:
        car.set_velocity(0, 90,  abs(yaw))


# ─────────────────────────────────────────────────────────────────────────────
# LED / buzzer helpers  (from obstacle avoidance script, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def set_led_color(name):
    try:
        colors = {
            'green':  [1, 0, 255, 0],
            'red':    [1, 255, 0, 0],
            'yellow': [1, 255, 150, 0],
            'blue':   [1, 0, 0, 255],
        }
        entry = colors.get(name, [1, 0, 0, 0])
        board.set_rgb([entry, [entry[0] + 1] + entry[1:]])
    except Exception:
        pass

def beep_once():
    try:
        board.set_buzzer(1900, 0.08, 0.05, 1)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Sonar helpers  (from obstacle avoidance script, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class DistanceFilter:
    def __init__(self, window=DIST_WINDOW):
        self.window  = window
        self.samples = []

    def update(self, dist_cm):
        self.samples.append(dist_cm)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        return float(np.mean(self.samples))

def read_distance_cm():
    return sonar.getDistance() / 10.0

def obstacle_detected(filtered_cm, threshold_cm=SAFE_DISTANCE_CM):
    return filtered_cm <= threshold_cm


# ─────────────────────────────────────────────────────────────────────────────
# Avoidance maneuver  (from obstacle avoidance script, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def avoid_obstacle():
    global last_avoid_time

    set_led_color('red')
    beep_once()

    if DEBUG_PRINT:
        print('Obstacle detected -> avoidance maneuver starting')

    stop_car()
    time.sleep(STOP_PAUSE)

    if DEBUG_PRINT:
        print('  reversing')
    drive_reverse(REVERSE_SPEED)
    time.sleep(REVERSE_TIME)

    if DEBUG_PRINT:
        print(f'  turning {TURN_DIRECTION}')
    turn_in_place(TURN_DIRECTION, TURN_YAW)
    time.sleep(TURN_TIME)

    if DEBUG_PRINT:
        print('  recovery forward')
    drive_forward(FORWARD_SPEED)
    time.sleep(FORWARD_RECOVERY_TIME)

    stop_car()
    time.sleep(0.10)

    last_avoid_time = time.time()
    set_led_color('yellow')

    if DEBUG_PRINT:
        print('Avoidance maneuver complete')


# ─────────────────────────────────────────────────────────────────────────────
# Color detection helpers  (from color_detection.py, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def apply_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask

def largest_contour_area(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return int(cv2.contourArea(max(contours, key=cv2.contourArea)))

def draw_overlay(frame: np.ndarray, color: str, areas: dict) -> np.ndarray:
    overlay     = frame.copy()
    label_color = {"red": (0, 0, 255), "yellow": (0, 220, 255),
                   "green": (0, 200, 0), "none": (180, 180, 180)}
    cv2.putText(overlay, f"Detected: {color.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                label_color.get(color, (255, 255, 255)), 2)
    for c, area in areas.items():
        if area > MIN_CONTOUR_AREA:
            cv2.putText(overlay, f"{c}: {area}px",
                        (10, 60 + list(areas).index(c) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        label_color.get(c, (200, 200, 200)), 1)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Vision Thread  (color_detection.py logic — detection only, no motor calls)
# The motor control is handled entirely by the main loop so that the sonar
# avoidance maneuver always takes priority.
# ─────────────────────────────────────────────────────────────────────────────

def vision_thread() -> None:
    global detected_color, shared_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    if not cap.isOpened():
        print("[Vision] ERROR: cannot open camera — exiting vision thread")
        stop_event.set()
        return

    confirm_counts  = {c: 0 for c in COLOR_RANGES}
    last_event_time = 0.0

    print("[Vision] thread started")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv  = cv2.GaussianBlur(hsv, (7, 7), 0)           # same blur as color_detection.py

        areas = {c: largest_contour_area(apply_mask(hsv, c)) for c in COLOR_RANGES}

        dominant = max(areas, key=areas.get)
        if areas[dominant] < MIN_CONTOUR_AREA:
            dominant = "none"

        # Confirmation hysteresis — same as color_detection.py
        for c in COLOR_RANGES:
            confirm_counts[c] = confirm_counts[c] + 1 if c == dominant else 0

        confirmed = "none"
        for c in COLOR_RANGES:
            if confirm_counts[c] >= DETECTION_CONFIRM:
                confirmed = c
                break

        # Debounce — only update shared state if enough time has passed
        now = time.time()
        if confirmed != "none" and (now - last_event_time) >= COLOR_DEBOUNCE_S:
            last_event_time = now

        with color_lock:
            detected_color = confirmed

        annotated = draw_overlay(frame, confirmed, areas)
        with frame_lock:
            shared_frame = annotated

    cap.release()
    print("[Vision] thread exited")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup / signal handler  (from obstacle avoidance script)
# ─────────────────────────────────────────────────────────────────────────────

def cleanup():
    global running
    running = False
    stop_event.set()
    stop_car()
    set_led_color('off')
    try:
        sonar.setRGBMode(0)
        sonar.setPixelColor(0, (0, 0, 0))
        sonar.setPixelColor(1, (0, 0, 0))
    except Exception:
        pass

def handle_signal(signum, frame):
    print('\nStopping...')
    cleanup()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop  (obstacle avoidance script's main loop + color priority layer)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global running, last_distance_cm

    signal.signal(signal.SIGINT, handle_signal)

    sonar.setRGBMode(0)
    set_led_color('blue')

    dist_filter = DistanceFilter(DIST_WINDOW)

    # Start vision thread
    vt = threading.Thread(target=vision_thread, name="VisionThread", daemon=True)
    vt.start()

    running = True
    print('TurboPi combined avoidance started')
    print(f'SAFE_DISTANCE_CM = {SAFE_DISTANCE_CM:.1f} cm')
    print(f'TURN_DIRECTION   = {TURN_DIRECTION}')
    print('Press Ctrl+C to stop\n')

    while running:
        # ── Sonar read (from obstacle avoidance script) ───────────────────────
        raw_distance      = read_distance_cm()
        filtered_distance = dist_filter.update(raw_distance)
        last_distance_cm  = filtered_distance

        if DEBUG_PRINT:
            print(f'Dist: {filtered_distance:5.1f} cm', end='  ')

        # ── Read latest color from vision thread ──────────────────────────────
        with color_lock:
            color = detected_color

        if DEBUG_PRINT:
            print(f'Color: {color.upper()}')

        now          = time.time()
        cooldown_done = (now - last_avoid_time) > COOLDOWN_TIME

        # ── Priority 1: sonar obstacle → run avoidance maneuver ───────────────
        if cooldown_done and obstacle_detected(filtered_distance, SAFE_DISTANCE_CM):
            avoid_obstacle()

        # ── Priority 2: traffic RED → stop ────────────────────────────────────
        elif color == 'red':
            set_led_color('red')
            stop_car()

        # ── Priority 3: traffic YELLOW → half speed ───────────────────────────
        elif color == 'yellow':
            set_led_color('yellow')
            drive_forward(FORWARD_SPEED // 2)

        # ── Priority 4: GREEN or no signal → full forward ─────────────────────
        else:
            set_led_color('green')
            drive_forward(FORWARD_SPEED)

        # ── Optional display ──────────────────────────────────────────────────
        with frame_lock:
            frame = shared_frame
        if frame is not None:
            cv2.imshow("Combined Avoidance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(LOOP_DELAY)

    cleanup()
    cv2.destroyAllWindows()
    vt.join(timeout=3.0)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup()
