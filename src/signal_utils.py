import os
import re
import subprocess
from datetime import UTC, datetime

SIGNAL_LOG_PATH = "~/Library/Application Support/Signal/logs/app.log"

import time

import Quartz


def get_signal_window_bounds():
    script = """
    tell application "System Events"
        tell process "Signal"
            set {x, y} to position of window 1
            set {w, h} to size of window 1
            return (x as string) & "," & (y as string) & "," & (w as string) & "," & (h as string)
        end tell
    end tell
    """
    out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
    return tuple(map(float, out.split(",")))


def hardware_click(x: float, y: float):
    point = (x, y)
    # Move cursor first so hover state registers in Electron
    move = Quartz.CGEventCreateMouseEvent(  # ty: ignore[missing-attribute]
        None,
        Quartz.kCGEventMouseMoved,  # ty: ignore[missing-attribute]
        point,
        Quartz.kCGMouseButtonLeft,  # ty: ignore[missing-attribute]
    )
    down = Quartz.CGEventCreateMouseEvent(  # ty: ignore[missing-attribute]
        None,
        Quartz.kCGEventLeftMouseDown,  # ty: ignore[missing-attribute]
        point,
        Quartz.kCGMouseButtonLeft,  # ty: ignore[missing-attribute]
    )
    up = Quartz.CGEventCreateMouseEvent(  # ty: ignore[missing-attribute]
        None,
        Quartz.kCGEventLeftMouseUp,  # ty: ignore[missing-attribute]
        point,
        Quartz.kCGMouseButtonLeft,  # ty: ignore[missing-attribute]
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move)  # ty: ignore[missing-attribute]
    time.sleep(0.05)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)  # ty: ignore[missing-attribute]
    time.sleep(0.05)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)  # ty: ignore[missing-attribute]


def make_call():
    # Use AppleScript and Quartz to perform the following actions:
    # 1. Activate the Signal app
    # 2. Select first contact by pressing `Cmd+1`
    # 3. Input `Ctrl+Shift+C` to open the call window
    # 4. Click the "Call" button in the call window
    applescript = """
    tell application "Signal"
        activate
    end tell
    delay 0.8

    tell application "System Events"
        tell process "Signal"
            set frontmost to true
            key code 18 using {command down} -- Cmd+1
            delay 0.6
            keystroke "c" using {control down, shift down}
        end tell
    end tell
    """
    subprocess.run(["osascript", "-e", applescript], check=True)
    time.sleep(1.0)

    x, y, w, h = get_signal_window_bounds()
    start_x = x + (w * 0.74)
    start_y = y + (h * 0.92)
    hardware_click(start_x, start_y)


def answer_call():
    # Use AppleScript and Quartz to perform the following actions:
    # 1. Activate the Signal app
    # 2. Click `Cmd+Shift+A` to answer the call
    applescript = """
    tell application "Signal" to activate

    delay 0.2

    tell application "System Events"
        if process "Signal" exists then
            tell process "Signal"
                -- Directly dispatch the shortcut without strictly needing window focus validation
                key code 0 using {control down, shift down}
            end tell
        end if
    end tell
    """
    subprocess.run(["osascript", "-e", applescript], check=True)


# regexp for for fetching datetime from log line: {"level":30,"time":"2026-11-19T22:23:57.166Z","msg":"[notifications] NotificationService: showing a notification IncomingCall [object Undefined]"}
datetime_regexp = r'"time":"([^"]+)"'


def get_datetime_from_log_line(line: str) -> datetime:
    match = re.search(datetime_regexp, line)
    if match is not None:
        return datetime.fromisoformat(match.group(1))
    else:
        raise ValueError(f"Could not find datetime in the log line: '{line}'")


def get_last_log_line_datetime(lines: list[str] | None = None) -> datetime:
    try:
        if lines is None:
            with open(os.path.expanduser(SIGNAL_LOG_PATH), "r") as f:
                lines = f.readlines()
        if not lines:
            return datetime.min.replace(tzinfo=UTC)
        last_line = lines[-1]
        return get_datetime_from_log_line(last_line)
    except FileNotFoundError:
        raise FileNotFoundError(f"Signal log file not found at '{SIGNAL_LOG_PATH}'.")


def clear_log_file():
    with open(os.path.expanduser(SIGNAL_LOG_PATH), "w") as f:
        f.write("")


def detect_incoming_call(
    last_answered_datetime: datetime | None,
    final_line_datetime_for_last_check: datetime | None,
) -> tuple[datetime | None, datetime]:
    """
    Detects if there is an incoming call in Signal by checking the log file for
    the "IncomingCall" notification. Returns True if an incoming call is detected
    and it has not been answered yet.
    """

    with open(os.path.expanduser(SIGNAL_LOG_PATH), "r") as f:
        lines = f.readlines()

    final_line_datetime = get_last_log_line_datetime(lines)

    # If final_line_datetime_for_last_check is provided, filter lines to only those after that timestamp
    if final_line_datetime_for_last_check:
        lines = [
            line
            for line in lines
            if get_datetime_from_log_line(line) > final_line_datetime_for_last_check
        ]

    # If last_answered_datetime is provided, and it is after the last log line's datetime, we can skip checking for incoming calls
    if last_answered_datetime:
        if not final_line_datetime_for_last_check:
            raise ValueError(
                "final_line_datetime_for_last_check must be provided if last_answered_datetime is provided"
            )
        if last_answered_datetime > final_line_datetime_for_last_check:
            print(
                f"Last answered call datetime '{last_answered_datetime.isoformat()}' is after the last log line datetime '{final_line_datetime.isoformat()}'. Skipping incoming call detection."
            )
            return None, final_line_datetime

    # Check for incoming call notifications in the filtered lines
    # Search for '[notifications] NotificationService: showing a notification IncomingCall [object Undefined]'
    for line in lines:
        if (
            "[notifications] NotificationService: showing a notification IncomingCall [object Undefined]"
            in line
        ):
            print(f"Incoming call detected in log line: {line.strip()}")

            answered_datetime = get_datetime_from_log_line(line)

            if last_answered_datetime and answered_datetime <= last_answered_datetime:
                print(
                    f"Incoming call detected, but a call was already answered at {last_answered_datetime}. Ignoring."
                )
                return None, final_line_datetime

            return answered_datetime, final_line_datetime

    return None, final_line_datetime


def detect_terminated_call():
    """
    Detects if a call has been terminated in Signal by checking the log file for
    the "CallTerminated" notification. Returns True if a call termination is detected.
    """
    with open(os.path.expanduser(SIGNAL_LOG_PATH), "r") as f:
        lines = f.readlines()

    # Check for call termination notifications in the lines
    # Search for 'terminate_active_call()'
    for line in lines:
        if "terminate_active_call()" in line:
            print(f"Call termination detected in log line: {line.strip()}")
            return True

    return False
