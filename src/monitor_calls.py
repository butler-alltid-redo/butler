import time

from signal_utils import (
    answer_call,
    clear_log_file,
    detect_incoming_call,
    detect_terminated_call,
    get_last_log_line_datetime,
)


def main():
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

            # TODO: Start an agent session with STT+LLM+TTS. End the session if the call is terminated.

            # Check if call is terminated
            while True:
                if detect_terminated_call():
                    print("Call terminated.")
                    break
                time.sleep(1)  # Check for call termination every second

        time.sleep(1)


if __name__ == "__main__":
    main()
