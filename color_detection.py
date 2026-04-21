#!/usr/bin/env python3
"""
Color Detection Module — HiWonder TurboPi
Traffic light color detection (Red / Yellow / Green) using HSV thresholding.

Thread architecture (matches CDR):
  Vision Thread
      └─ spawns Motor Control Thread on each color-change event
             └─ Vision Thread joins Motor Control Thread, then loops
"""

import cv2
import numpy as np
import threading
import time
import sys

# ── TurboPi SDK ──────────────────────────────────────────────────────────────
try:
    from hiwonder.TurboPi import TurboPi
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("[WARN] TurboPi SDK not found — running in simulation mode")

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
CAMERA_FPS     = 30

# ── Detection tuning ─────────────────────────────────────────────────────────
MIN_CONTOUR_AREA  = 1500   # px² — ignore blobs smaller than this
DETECTION_CONFIRM = 3      # consecutive frames required before acting
COLOR_DEBOUNCE_S  = 0.4    # seconds — minimum time between color-change events

# ── HSV color ranges ──────────────────────────────────────────────────────────
# Red wraps around H=0/180 in OpenCV HSV, so two ranges are needed.
# Yellow calibration was flagged as missing in the CDR — adjust
# YELLOW_LOWER/UPPER on the physical car under actual lighting.
COLOR_RANGES = {
    "red": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),
    ],
    "yellow": [
        # TODO: calibrate under actual track lighting (CDR issue #3)
        (np.array([18,  100,  80]),  np.array([35,  255, 255])),
    ],
    "green": [
        (np.array([36,   80,  60]),  np.array([89,  255, 255])),
    ],
}

# ── Motor command mapping ─────────────────────────────────────────────────────
# Duration in seconds each command runs before the motor thread exits.
# Red   → stop (obey traffic light)
# Yellow → slow to half speed as caution
# Green  → full forward
MOTOR_COMMANDS = {
    "red":    {"speed": 0,   "direction": 90, "angular": 0, "duration": 2.0},
    "yellow": {"speed": 30,  "direction": 90, "angular": 0, "duration": 1.0},
    "green":  {"speed": 60,  "direction": 90, "angular": 0, "duration": 1.0},
    "none":   {"speed": 50,  "direction": 90, "angular": 0, "duration": 0.5},
}

# ── Shared state ──────────────────────────────────────────────────────────────
detected_color:   str   = "none"
shared_frame:     None  = None
frame_lock              = threading.Lock()
stop_event              = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Union of all HSV ranges for a given color label."""
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
    """Draw detected color label and bounding-box preview on frame."""
    overlay = frame.copy()
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
# Motor Control Thread
# ─────────────────────────────────────────────────────────────────────────────

def motor_control_thread(color: str, robot) -> None:
    """
    Executes the motor command corresponding to the detected traffic light
    color, then exits so the Vision Thread can join it.
    """
    cmd = MOTOR_COMMANDS.get(color, MOTOR_COMMANDS["none"])
    print(f"[Motor] {color.upper()} → speed={cmd['speed']}  dur={cmd['duration']}s")

    if ROBOT_AVAILABLE and robot is not None:
        robot.set_velocity(cmd["speed"], cmd["direction"], cmd["angular"])
        time.sleep(cmd["duration"])
        robot.set_velocity(0, 90, 0)   # halt after command window
    else:
        time.sleep(cmd["duration"])    # simulate timing in headless mode

    print(f"[Motor] command complete for {color.upper()}")


# ─────────────────────────────────────────────────────────────────────────────
# Vision Thread  (main detection loop)
# ─────────────────────────────────────────────────────────────────────────────

def vision_thread(robot) -> None:
    """
    Captures frames, detects dominant traffic-light color, and spawns a
    Motor Control Thread whenever the detected color changes.

    Confirmation hysteresis (DETECTION_CONFIRM frames) and a debounce timer
    (COLOR_DEBOUNCE_S) guard against rapid red↔yellow oscillation (CDR §next-steps).
    """
    global detected_color, shared_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    if not cap.isOpened():
        print("[Vision] ERROR: cannot open camera")
        stop_event.set()
        return

    confirm_counts  = {c: 0 for c in COLOR_RANGES}
    last_event_time = 0.0
    active_motor_t  = None

    print("[Vision] thread started")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Vision] frame read failed — retrying")
            time.sleep(0.05)
            continue

        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Mild Gaussian blur reduces noise before thresholding
        hsv   = cv2.GaussianBlur(hsv, (7, 7), 0)

        areas = {c: largest_contour_area(apply_mask(hsv, c)) for c in COLOR_RANGES}

        # ── Dominant color: largest blob above MIN_CONTOUR_AREA ──────────────
        dominant = max(areas, key=areas.get)
        if areas[dominant] < MIN_CONTOUR_AREA:
            dominant = "none"

        # ── Confirmation hysteresis ───────────────────────────────────────────
        for c in COLOR_RANGES:
            confirm_counts[c] = confirm_counts[c] + 1 if c == dominant else 0

        confirmed_color = "none"
        for c in COLOR_RANGES:
            if confirm_counts[c] >= DETECTION_CONFIRM:
                confirmed_color = c
                break

        # ── Update shared state ───────────────────────────────────────────────
        with frame_lock:
            detected_color = confirmed_color
            shared_frame   = draw_overlay(frame, confirmed_color, areas)

        # ── Spawn Motor Control Thread on confirmed color change ───────────────
        now = time.time()
        color_changed   = confirmed_color != "none"
        debounce_passed = (now - last_event_time) >= COLOR_DEBOUNCE_S
        motor_free      = (active_motor_t is None or not active_motor_t.is_alive())

        if color_changed and debounce_passed and motor_free:
            last_event_time = now
            active_motor_t  = threading.Thread(
                target=motor_control_thread,
                args=(confirmed_color, robot),
                daemon=True,
                name=f"MotorCtrl-{confirmed_color}"
            )
            active_motor_t.start()
            # CDR: Vision Thread waits for Motor Control Thread to exit
            active_motor_t.join()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if active_motor_t and active_motor_t.is_alive():
        active_motor_t.join(timeout=3.0)

    cap.release()
    print("[Vision] thread exited")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization (optional — display loop on the host machine)
# ─────────────────────────────────────────────────────────────────────────────

def visualization_loop() -> None:
    """
    Reads shared_frame and displays it; press 'q' to quit.
    Can be omitted on headless deployments — remove the call in main().
    """
    print("[Viz] press 'q' to quit")
    while not stop_event.is_set():
        with frame_lock:
            frame = shared_frame

        if frame is not None:
            cv2.imshow("Color Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[Viz] visualization closed")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    robot = None
    if ROBOT_AVAILABLE:
        robot = TurboPi()
        robot.set_velocity(0, 90, 0)   # ensure motors start stopped

    # Launch Vision Thread
    vt = threading.Thread(target=vision_thread, args=(robot,), name="VisionThread", daemon=True)
    vt.start()

    # Visualization runs on the main thread (requires display); comment out for headless
    try:
        visualization_loop()
    except KeyboardInterrupt:
        print("\n[Main] keyboard interrupt — stopping")
        stop_event.set()

    vt.join(timeout=5.0)

    if ROBOT_AVAILABLE and robot is not None:
        robot.set_velocity(0, 90, 0)

    print("[Main] color detection exited cleanly")


if __name__ == "__main__":
    main()
