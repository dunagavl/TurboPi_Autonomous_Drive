#!/usr/bin/python3
# coding=utf8

"""
TurboPi obstacle avoidance demo with FSM + sweeping cruise motion.

Behavior:
1. Cruise forward in a gentle sweeping motion
2. Continuously read the front ultrasonic sensor
3. If an obstacle is detected within the safe distance:
   - reverse
   - turn away
   - move forward to recover
4. Resume sweeping cruise

Why sweeping helps:
- A single front-facing sonar only sees what is roughly in front of it
- By gently sweeping the car while moving forward, the sonar samples a wider area
- This improves detection of obstacles that are slightly off-center

Notes:
- No ROS is used
- No IMU is used
"""

import sys
sys.path.append('/home/pi/TurboPi/')

import time
import math
import signal
import numpy as np
from enum import Enum, auto

import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc


if sys.version_info.major == 2:
    print('Please run this script with python3!')
    sys.exit(0)


# ============================================================
# USER PARAMETERS
# ============================================================

# Motion settings
FORWARD_SPEED = 35
REVERSE_SPEED = 30
TURN_YAW = 0.45
TURN_DIRECTION = 'left'      # 'left' or 'right'

# Sweeping cruise settings
SWEEP_YAW_AMPLITUDE = 0.20   # how hard the robot weaves left/right
SWEEP_PERIOD = 1.80          # seconds for one full left-right-left sweep

# Sonar settings
SAFE_DISTANCE_CM = 24.0
DIST_WINDOW = 5

# FSM timing
LOOP_DELAY = 0.05
REVERSE_TIME = 0.40
TURN_TIME = 0.65
FORWARD_RECOVERY_TIME = 0.45

# Debug
DEBUG_PRINT = True
DEBUG_PRINT_PERIOD = 0.20


# ============================================================
# GLOBAL OBJECTS
# ============================================================
car = mecanum.MecanumChassis()
sonar = Sonar.Sonar()
board = rrc.Board()

running = False
last_distance_cm = 999.0


# ============================================================
# FSM STATES
# ============================================================
class State(Enum):
    CRUISE = auto()
    REVERSE = auto()
    TURN = auto()
    RECOVER = auto()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def set_led_color(name):
    """Set onboard RGB LEDs to show current state."""
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


def beep_once():
    """Short buzzer beep for transition/debug feedback."""
    try:
        board.set_buzzer(1900, 0.08, 0.05, 1)
    except Exception:
        pass


def read_distance_cm():
    """
    Read ultrasonic distance in centimeters.

    TurboPi sonar.getDistance() returns millimeters.
    """
    raw_mm = sonar.getDistance()
    return raw_mm / 10.0


def cleanup():
    """Safe shutdown."""
    global running
    running = False

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
    print('\nStopping obstacle avoidance demo...')
    cleanup()
    sys.exit(0)


# ============================================================
# FILTER
# ============================================================
class DistanceFilter:
    """Simple moving average filter for sonar readings."""
    def __init__(self, window=DIST_WINDOW):
        self.window = window
        self.samples = []

    def update(self, dist_cm):
        self.samples.append(dist_cm)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        return float(np.mean(self.samples))


# ============================================================
# MOTION LAYER
# ============================================================
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
        """
        Drive forward while smoothly oscillating yaw.

        yaw(t) = amplitude * sin(2*pi*t/period)

        This causes a gentle left-right sweeping path so the front sonar
        samples more than just the dead-ahead direction.
        """
        yaw = amplitude * math.sin((2.0 * math.pi * t_in_state) / period)
        self.command = 'sweep_forward'
        self.last_yaw = yaw
        self.car.set_velocity(speed, 90, yaw)
        return yaw


# ============================================================
# FSM
# ============================================================
class RobotFSM:
    def __init__(self, motion):
        self.motion = motion
        self.state = State.CRUISE
        self.state_start_time = time.time()
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

    def update(self, distance_cm):
        if self.state == State.CRUISE:
            self.motion.drive_forward_sweep(
                t_in_state=self.time_in_state(),
                speed=FORWARD_SPEED,
                amplitude=SWEEP_YAW_AMPLITUDE,
                period=SWEEP_PERIOD
            )

            if distance_cm <= SAFE_DISTANCE_CM:
                if DEBUG_PRINT:
                    print('Sonar obstacle detected -> REVERSE')
                self.set_state(State.REVERSE)
                return

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
                self.set_state(State.CRUISE)
                return


# ============================================================
# MAIN DEMO
# ============================================================
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

    running = True
    last_debug_time = 0.0

    print('TurboPi obstacle avoidance FSM demo started')
    print(f'SAFE_DISTANCE_CM      = {SAFE_DISTANCE_CM:.1f} cm')
    print(f'TURN_DIRECTION        = {TURN_DIRECTION}')
    print(f'SWEEP_YAW_AMPLITUDE   = {SWEEP_YAW_AMPLITUDE}')
    print(f'SWEEP_PERIOD          = {SWEEP_PERIOD:.2f} s')
    print('Press Ctrl+C to stop\n')

    while running:
        raw_distance = read_distance_cm()
        filtered_distance = dist_filter.update(raw_distance)
        last_distance_cm = filtered_distance

        fsm.update(filtered_distance)

        now = time.time()
        if DEBUG_PRINT and (now - last_debug_time) >= DEBUG_PRINT_PERIOD:
            print(
                f'State={fsm.state.name:8s} | '
                f'RawDist={raw_distance:5.1f} cm | '
                f'FiltDist={filtered_distance:5.1f} cm | '
                f'Cmd={motion.command:13s} | '
                f'Yaw={motion.last_yaw:6.3f}'
            )
            last_debug_time = now

        time.sleep(LOOP_DELAY)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup()