#!/usr/bin/env python3
# coding=utf8

"""
Combined Avoidance and Color Detection  HiWonder TurboPi

This version integrates:
  the color detection prototype structure
  the newer sweep-based obstacle avoidance behavior

Architecture:
  VisionThread -   camera loop; updates detected_color + annotated frame
  Main loop   -    reads sonar + color; owns all motor commands

Priority:
  1. Sonar obstacle    -> avoidance FSM (REVERSE / TURN / RECOVER)
  2. Traffic RED       -> stop
  3. Traffic YELLOW    -> slow sweeping forward motion
  4. Traffic GREEN     -> normal sweeping forward motion
  5. No signal         -> normal sweeping forward motion
"""

import sys
sys.path.append('/home/pi/TurboPi/')

import cv2
import math
import numpy as np
import threading
import time
import signal
from enum import Enum, auto

import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this script with python3!')
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware objects
# ─────────────────────────────────────────────────────────────────────────────
car = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
board = rrc.Board()


# ─────────────────────────────────────────────────────────────────────────────
# Motion / avoidance parameters
# ─────────────────────────────────────────────────────────────────────────────
FORWARD_SPEED = 35
YELLOW_SPEED = 18
REVERSE_SPEED = 30

SAFE_DISTANCE_CM = 35.0
DIST_WINDOW = 5
LOOP_DELAY = 0.05

REVERSE_TIME = 0.40
TURN_TIME = 0.65
FORWARD_RECOVERY_TIME = 0.45
COOLDOWN_TIME = 1.00

TURN_DIRECTION = 'left'
TURN_YAW = 0.45

# Sweeping cruise motion
SWEEP_YAW_AMPLITUDE = 0.35
YELLOW_SWEEP_YAW_AMPLITUDE = 0.12
SWEEP_PERIOD = 1.80

DEBUG_PRINT = True
DEBUG_PRINT_PERIOD = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Camera parameters
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Color detection parameters
# ─────────────────────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA = 1500
DETECTION_CONFIRM = 3
COLOR_DEBOUNCE_S = 0.4

COLOR_RANGES = {
    "red": [
        (np.array([0,   120,  70]), np.array([10,  255, 255])),
        (np.array([170, 120,  70]), np.array([180, 255, 255])),
    ],
    "yellow": [
        (np.array([18, 100, 80]), np.array([35, 255, 255])),
    ],
    "green": [
        (np.array([36, 80, 60]), np.array([89, 255, 255])),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared thread state
# ─────────────────────────────────────────────────────────────────────────────
detected_color = "none"
shared_frame = None

color_lock = threading.Lock()
frame_lock = threading.Lock()
stop_event = threading.Event()

running = True
last_distance_cm = 999.0


# ─────────────────────────────────────────────────────────────────────────────
# FSM states
# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    CRUISE = auto()
    REVERSE = auto()
    TURN = auto()
    RECOVER = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Motor / board helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_led_color(name):
    try:
        if name == 'green':
            board.set_rgb([[1, 0, 255, 0], [2, 0, 255, 0]])
        elif name == 'red':
            board.set_rgb([[1, 255, 0, 0], [2, 255, 0, 0]])
        elif name == 'yellow':
            board.set_rgb([[1, 255, 150, 0], [2, 255, 150, 0]])
        elif name == 'blue':
            board.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])
        elif name == 'purple':
            board.set_rgb([[1, 255, 0, 255], [2, 255, 0, 255]])
        else:
            board.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
    except Exception:
        pass


class MotionController:
    def __init__(self, car):
        self.car = car
        self.command = 'stop'
        self.last_yaw = 0.0

    def stop_car(self):
        self.command = 'stop'
        self.last_yaw = 0.0
        self.car.set_velocity(0, 90, 0)

    def drive_forward(self, speed=FORWARD_SPEED):
        self.command = 'forward'
        self.last_yaw = 0.0
        self.car.set_velocity(speed, 90, 0)

    def drive_reverse(self, speed=REVERSE_SPEED):
        self.command = 'reverse'
        self.last_yaw = 0.0
        self.car.set_velocity(speed, 270, 0)

    def turn_in_place(self, direction='left', yaw=TURN_YAW):
        self.command = f'turn_{direction}'
        if direction == 'left':
            self.last_yaw = -abs(yaw)
            self.car.set_velocity(0, 90, self.last_yaw)
        else:
            self.last_yaw = abs(yaw)
            self.car.set_velocity(0, 90, self.last_yaw)

    def drive_forward_sweep(self, t_in_state, speed=FORWARD_SPEED,
                            amplitude=SWEEP_YAW_AMPLITUDE, period=SWEEP_PERIOD):
        yaw = amplitude * math.sin((2.0 * math.pi * t_in_state) / period)
        self.command = 'sweep_forward'
        self.last_yaw = yaw
        self.car.set_velocity(speed, 90, yaw)
        return yaw


# ─────────────────────────────────────────────────────────────────────────────
# Sonar helpers
# ─────────────────────────────────────────────────────────────────────────────
class DistanceFilter:
    def __init__(self, window=DIST_WINDOW):
        self.window = window
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
# Avoidance FSM
# ─────────────────────────────────────────────────────────────────────────────
class RobotFSM:
    def __init__(self, motion):
        self.motion = motion
        self.state = State.CRUISE
        self.state_start_time = time.time()
        self.last_avoid_end_time = 0.0
        set_led_color('green')

    def set_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()

        if DEBUG_PRINT:
            print(f'--> Entering state: {self.state.name}')

        if new_state == State.CRUISE:
            set_led_color('green')

        elif new_state == State.REVERSE:
            set_led_color('red')

        elif new_state == State.TURN:
            set_led_color('blue')

        elif new_state == State.RECOVER:
            set_led_color('purple')

    def time_in_state(self):
        return time.time() - self.state_start_time

    def update(self, distance_cm, color_name):
        now = time.time()

        if self.state == State.CRUISE:
            cooldown_done = (now - self.last_avoid_end_time) >= COOLDOWN_TIME

            # Highest priority: sonar obstacle
            if cooldown_done and obstacle_detected(distance_cm, SAFE_DISTANCE_CM):
                if DEBUG_PRINT:
                    print('Sonar obstacle detected -> REVERSE')
                self.set_state(State.REVERSE)
                return

            # Otherwise obey traffic-light color
            if color_name == 'red':
                set_led_color('red')
                self.motion.stop_car()

            elif color_name == 'yellow':
                set_led_color('yellow')
                self.motion.drive_forward_sweep(
                    t_in_state=self.time_in_state(),
                    speed=YELLOW_SPEED,
                    amplitude=YELLOW_SWEEP_YAW_AMPLITUDE,
                    period=SWEEP_PERIOD
                )

            else:
                set_led_color('green')
                self.motion.drive_forward_sweep(
                    t_in_state=self.time_in_state(),
                    speed=FORWARD_SPEED,
                    amplitude=SWEEP_YAW_AMPLITUDE,
                    period=SWEEP_PERIOD
                )

        elif self.state == State.REVERSE:
            self.motion.drive_reverse(REVERSE_SPEED)
            if self.time_in_state() >= REVERSE_TIME:
                self.set_state(State.TURN)
                return

        elif self.state == State.TURN:
            self.motion.turn_in_place(TURN_DIRECTION, TURN_YAW)
            if self.time_in_state() >= TURN_TIME:
                self.set_state(State.RECOVER)
                return

        elif self.state == State.RECOVER:
            self.motion.drive_forward(FORWARD_SPEED)
            if self.time_in_state() >= FORWARD_RECOVERY_TIME:
                self.last_avoid_end_time = time.time()
                self.set_state(State.CRUISE)
                return


# ─────────────────────────────────────────────────────────────────────────────
# Color detection helpers
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


def draw_overlay(frame: np.ndarray, color: str, areas: dict,
                 fsm_state: str = "", distance_cm: float = -1.0) -> np.ndarray:
    overlay = frame.copy()
    label_color = {
        "red": (0, 0, 255),
        "yellow": (0, 220, 255),
        "green": (0, 200, 0),
        "none": (180, 180, 180)
    }

    cv2.putText(
        overlay,
        f"Detected: {color.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        label_color.get(color, (255, 255, 255)),
        2
    )

    if fsm_state:
        cv2.putText(
            overlay,
            f"FSM: {fsm_state}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    if distance_cm >= 0.0:
        cv2.putText(
            overlay,
            f"Dist: {distance_cm:.1f} cm",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    row = 120
    for c, area in areas.items():
        if area > MIN_CONTOUR_AREA:
            cv2.putText(
                overlay,
                f"{c}: {area}px",
                (10, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                label_color.get(c, (200, 200, 200)),
                1
            )
            row += 25

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Vision thread
# ─────────────────────────────────────────────────────────────────────────────
def vision_thread() -> None:
    global detected_color, shared_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        print("[Vision] ERROR: cannot open camera - exiting vision thread")
        stop_event.set()
        return

    confirm_counts = {c: 0 for c in COLOR_RANGES}
    last_event_time = 0.0

    print("[Vision] thread started")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

        areas = {c: largest_contour_area(apply_mask(hsv, c)) for c in COLOR_RANGES}

        dominant = max(areas, key=areas.get)
        if areas[dominant] < MIN_CONTOUR_AREA:
            dominant = "none"

        for c in COLOR_RANGES:
            confirm_counts[c] = confirm_counts[c] + 1 if c == dominant else 0

        confirmed = "none"
        for c in COLOR_RANGES:
            if confirm_counts[c] >= DETECTION_CONFIRM:
                confirmed = c
                break

        now = time.time()
        if confirmed != "none" and (now - last_event_time) >= COLOR_DEBOUNCE_S:
            last_event_time = now

        with color_lock:
            detected_color = confirmed

        # Pull current state for annotation
        with color_lock:
            current_color = detected_color

        annotated = draw_overlay(frame, current_color, areas)

        with frame_lock:
            shared_frame = annotated

    cap.release()
    print("[Vision] thread exited")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup / signal handler
# ─────────────────────────────────────────────────────────────────────────────
def cleanup():
    global running
    running = False
    stop_event.set()

    try:
        car.set_velocity(0, 90, 0)
    except Exception:
        pass

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
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global running, last_distance_cm

    signal.signal(signal.SIGINT, handle_signal)

    try:
        sonar.setRGBMode(0)
        sonar.setPixelColor(0, (0, 0, 0))
        sonar.setPixelColor(1, (0, 0, 0))
    except Exception:
        pass

    set_led_color('blue')

    dist_filter = DistanceFilter(DIST_WINDOW)
    motion = MotionController(car)
    fsm = RobotFSM(motion)

    vt = threading.Thread(target=vision_thread, name="VisionThread", daemon=True)
    vt.start()

    running = True
    last_debug_time = 0.0

    print('TurboPi combined avoidance started')
    print(f'SAFE_DISTANCE_CM      = {SAFE_DISTANCE_CM:.1f} cm')
    print(f'TURN_DIRECTION        = {TURN_DIRECTION}')
    print(f'SWEEP_YAW_AMPLITUDE   = {SWEEP_YAW_AMPLITUDE}')
    print(f'SWEEP_PERIOD          = {SWEEP_PERIOD:.2f} s')
    print('Press Ctrl+C to stop\n')

    while running and not stop_event.is_set():
        raw_distance = read_distance_cm()
        filtered_distance = dist_filter.update(raw_distance)
        last_distance_cm = filtered_distance

        with color_lock:
            color = detected_color

        fsm.update(filtered_distance, color)

        now = time.time()
        if DEBUG_PRINT and (now - last_debug_time) >= DEBUG_PRINT_PERIOD:
            print(
                f'State={fsm.state.name:8s} | '
                f'Dist={filtered_distance:5.1f} cm | '
                f'Color={color.upper():6s} | '
                f'Cmd={motion.command:13s} | '
                f'Yaw={motion.last_yaw:6.3f}'
            )
            last_debug_time = now

        with frame_lock:
            frame = shared_frame

        if frame is not None:
            annotated = draw_overlay(frame, color, {}, fsm.state.name, filtered_distance)
            cv2.imshow("Combined Avoidance", annotated)

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