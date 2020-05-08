import json
import os
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class EventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print(event)
        with open("squash/generated_outputs/queue/queue.txt", "r") as f:
            data = f.read().strip()
        if len(data) == 0:
            return
        next_key = data.split("\n")[0]

        # Wait until the question filtering is completed
        if not os.path.exists("squash/generated_outputs/final/%s.json" % next_key):
            return

        # once it is complete, verify it is a valid json before purging it from the queue
        print("Cleaning up %s" % next_key)
        try:
            with open('squash/generated_outputs/final/%s.json' % next_key, 'r') as f:
                final_data = json.loads(f.read())
        except Exception as e:
            # This is the unlikely situation where the final/%s.json is not completely written
            # This error wont cause a problem on the next cycle
            print(e)
            print("Error while processing squash/generated_outputs/final/%s.json" % next_key)
            return

        # once the json has been satisfactorily checked, pop it from the queue file using sed
        command = "sed -i '1d' squash/generated_outputs/queue/queue.txt"
        print(subprocess.check_output(command, shell=True))


if __name__ == "__main__":
    path = "squash/generated_outputs/final"
    event_handler = EventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
