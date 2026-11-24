
import time
from datetime import datetime

from gmail_auto_router import process_inbox

CHECK_INTERVAL = 60

print("Email Monitoring Started")
print("Checking Gmail every 60 seconds...")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        print(
            f"\nChecking inbox at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

        try:
            process_inbox()
        except Exception as error:
            print("Monitoring error:", error)

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\nEmail Monitoring Stopped")

