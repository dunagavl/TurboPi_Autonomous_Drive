#!/usr/bin/python3
# coding=utf8
"""
TurboPi Obstacle Avoidance Demo

Standalone, ROS-free obstacle avoidance script for the Hiwonder TurboPi.
This uses the same library style as the TurboPi example code:
- HiwonderSDK.mecanum for chassis motion
- HiwonderSDK.Sonar for ultrasonic distance
- HiwonderSDK.ros_robot_controller_sdk for board RGB/buzzer helpers

Behavior:
1. Drive forward normally
2. Continuously read the ultrasonic sensor
3. If an obstacle is too close:
   - stop
   - reverse briefly
   - turn left for a fixed time
   - move forward briefly
4. Resume forward driving

Notes:
- No ROS is used here.
- The board module filename contains "ros" because of the vendor SDK, but this
  script does not use ROS nodes, topics, launches, or roscore.
- Start with a simple fixed left turn. Once this works, you can extend it.
"""

import sys
sys.path.append('/home/pi/TurboPi/')

import time
import signal
import numpy as np

import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc


if sys.version_info.major == 2:
    print('Please run this script with python3!')
    sys.exit(0)


# ============================================================
# USER PARAMETERS
# ============================================================
FORWARD_SPEED = 35               # normal forward speed
REVERSE_SPEED = 30               # reverse speed during avoidance
SAFE_DISTANCE_CM = 24.0          # trigger distance threshold
DIST_WINDOW = 5                  # moving-average window length
LOOP_DELAY = 0.05                # main loop period in seconds

STOP_PAUSE = 0.15                # short pause after stopping
REVERSE_TIME = 0.40              # how long to back up
TURN_TIME = 0.65                 # how long to turn
FORWARD_RECOVERY_TIME = 0.45     # move forward after turning
COOLDOWN_TIME = 1.00             # time before another avoidance can trigger

TURN_DIRECTION = 'left'          # 'left' or 'right'
TURN_YAW = 0.45                  # yaw command magnitude for turning

DEBUG_PRINT = True


# ============================================================
# GLOBAL OBJECTS
# ============================================================
car = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
board = rrc.Board()

running = False
last_avoid_time = 0.0
last_distance_cm = 999.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def set_led_color(name):
    """Set onboard RGB LEDs to show current behavior."""
    try:
        if name == 'green':
            board.set_rgb([[1, 0, 255, 0], [2, 0, 255, 0]])
        elif name == 'red':
            board.set_rgb([[1, 255, 0, 0], [2, 255, 0, 0]])
        elif name == 'yellow':
            board.set_rgb([[1, 255, 150, 0], [2, 255, 150, 0]])
        elif name == 'blue':
            board.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])
        else:
            board.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
    except Exception:
        pass


def beep_once():
    """Optional beep when an obstacle is detected."""
    try:
        board.set_buzzer(1900, 0.08, 0.05, 1)
    except Exception:
        pass


def stop_car():
    """Stop all chassis motion."""
    car.set_velocity(0, 90, 0)


def drive_forward(speed=FORWARD_SPEED):
    """Move forward.

    TurboPi uses set_velocity(speed, direction, yaw_rate).
    direction=90 corresponds to forward in the vendor examples.
    yaw_rate=0 means no turning.
    """
    car.set_velocity(speed, 90, 0)


def drive_reverse(speed=REVERSE_SPEED):
    """Move backward.

    direction=270 corresponds to reverse in the vendor examples.
    """
    car.set_velocity(speed, 270, 0)


def turn_in_place(direction='left', yaw=TURN_YAW):
    """Rotate in place using chassis yaw control."""
    if direction == 'left':
        car.set_velocity(0, 90, -abs(yaw))
    else:
        car.set_velocity(0, 90, abs(yaw))


def cleanup():
    """Safe shutdown."""
    global running
    running = False
    stop_car()
    set_led_color('off')
    try:
        sonar.setRGBMode(0)
        sonar.setPixelColor(0, (0, 0, 0))
        sonar.setPixelColor(1, (0, 0, 0))
    except Exception:
        pass


def handle_signal(signum, frame):
    """Catch Ctrl+C and stop safely."""
    print('\nStopping obstacle avoidance...')
    cleanup()
    sys.exit(0)


class DistanceFilter:
    """Simple moving-average filter for ultrasonic readings.

    The ultrasonic sensor can produce noisy readings. Averaging the most recent
    few values makes the avoidance trigger more stable.
    """
    def __init__(self, window=DIST_WINDOW):
        self.window = window
        self.samples = []

    def update(self, dist_cm):
        self.samples.append(dist_cm)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        return float(np.mean(self.samples))


# ============================================================
# SENSOR + DECISION FUNCTIONS
# ============================================================
def read_distance_cm():
    """Read front ultrasonic distance in centimeters.

    In the TurboPi examples, sonar.getDistance() returns millimeters,
    so divide by 10 to convert to centimeters.
    """
    raw_mm = sonar.getDistance()
    return raw_mm / 10.0


def obstacle_detected(filtered_distance_cm, threshold_cm=SAFE_DISTANCE_CM):
    """Return True when an obstacle is closer than the threshold."""
    return filtered_distance_cm <= threshold_cm


# ============================================================
# AVOIDANCE MANEUVER
# ============================================================
def avoid_obstacle():
    """Perform a simple fixed avoidance maneuver.

    Sequence:
    1. Stop
    2. Reverse a little to create space
    3. Turn away from the obstacle
    4. Move forward to clear the obstacle region
    5. Stop briefly and return control to main loop
    """
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


# ============================================================
# MAIN CONTROL LOOP
# ============================================================
def main():
    global running, last_distance_cm

    signal.signal(signal.SIGINT, handle_signal)

    sonar.setRGBMode(0)
    set_led_color('blue')

    dist_filter = DistanceFilter(DIST_WINDOW)

    running = True
    print('TurboPi obstacle avoidance started')
    print(f'SAFE_DISTANCE_CM = {SAFE_DISTANCE_CM:.1f} cm')
    print(f'TURN_DIRECTION   = {TURN_DIRECTION}')
    print('Press Ctrl+C to stop\n')

    while running:
        # Read and filter distance
        raw_distance = read_distance_cm()
        filtered_distance = dist_filter.update(raw_distance)
        last_distance_cm = filtered_distance

        if DEBUG_PRINT:
            print(f'Raw: {raw_distance:5.1f} cm | Filtered: {filtered_distance:5.1f} cm')

        # If not currently avoiding and obstacle is close, trigger maneuver
        now = time.time()
        cooldown_done = (now - last_avoid_time) > COOLDOWN_TIME

        if cooldown_done and obstacle_detected(filtered_distance, SAFE_DISTANCE_CM):
            avoid_obstacle()
        else:
            # Normal cruising state
            set_led_color('green')
            drive_forward(FORWARD_SPEED)

        time.sleep(LOOP_DELAY)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup()
