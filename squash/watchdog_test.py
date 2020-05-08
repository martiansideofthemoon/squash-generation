import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class EventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        import pdb; pdb.set_trace()
        print(event)


if __name__ == "__main__":
    path = "/home/kalpesh/squash-generation/squash/temp"
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
