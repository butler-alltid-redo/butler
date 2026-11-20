"""Monitor Signal logs and start the voice assistant for incoming calls."""

import threading
import time

from agent import AgentSession, agent, interrupt_assistant
from signal_utils import (
    answer_call,
    clear_log_file,
    detect_incoming_call,
    detect_terminated_call,
    get_last_log_line_datetime,
)


def monitor_call_termination(session):
    """Stop the active agent when Signal reports call termination."""
    stop_event = session.stop_event

    while not stop_event.is_set():
        if detect_terminated_call():
            print("Call terminated.")
            stop_event.set()
            interrupt_assistant(session)
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

            session = AgentSession()
            termination_thread = threading.Thread(
                target=monitor_call_termination,
                args=(session,),
                name="call-termination-monitor",
                daemon=True,
            )
            termination_thread.start()

            agent(session)

            print("Call ended. Resuming monitoring for incoming calls...")

        time.sleep(1)


if __name__ == "__main__":
    main()
