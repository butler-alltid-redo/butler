"""Monitor Signal logs and start the voice assistant for incoming calls."""

import time

from agent import agent
from signal_utils import (
    answer_call,
    clear_log_file,
    detect_incoming_call,
    detect_terminated_call,
    get_last_log_line_datetime,
)


def monitor_call_termination():
    """Placeholder monitor that returns when Signal reports call termination."""
    while True:
        if detect_terminated_call():
            print("Call terminated.")
            return
        time.sleep(1)


def main():
    """Poll Signal logs and answer incoming calls indefinitely."""
    last_answered_datetime = None
    final_line_datetime = get_last_log_line_datetime()  # Initialize final_line_datetime to the datetime of the last line in the log file upon startup

    print(f"Last Signal log line datetime: '{final_line_datetime.isoformat()}'")
    print("Monitoring Signal logs for incoming calls...")

    while True:
        answered_datetime, final_line_datetime = detect_incoming_call(
            last_answered_datetime, final_line_datetime
        )
        if answered_datetime is not None:
            print("Incoming call detected. Answering...")
            answer_call()
            clear_log_file()  # Clear the log file after answering the call to avoid duplicate detections
            print("Call answered. Log file cleared.")
            last_answered_datetime = answered_datetime

            # TODO If call is terminated, stop the agent and return to monitoring for incoming calls.
            # termination_thread = threading.Thread(
            #     target=monitor_call_termination,
            #     name="call-termination-monitor",
            #     daemon=True,
            # )
            # termination_thread.start()

            agent()

        time.sleep(1)


if __name__ == "__main__":
    main()
