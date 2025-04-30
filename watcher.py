from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import requests
import hashlib

WATCH_FILE = "new_order.json"
last_file_hash = None

def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_file_hash
        if event.src_path.endswith(WATCH_FILE):
            new_hash = file_hash(WATCH_FILE)
            if new_hash is None:
                return
            if new_hash == last_file_hash:
                return
            last_file_hash = new_hash

            print(f"{WATCH_FILE} updated! Sending predict-eta request...")
            try:
                response = requests.get("http://localhost:5000/predict-eta")
                if response.ok:
                    print("Response:", response.json())
                else:
                    print("Server error:", response.status_code, response.text)
            except Exception as e:
                print("Error calling server:", e)

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(FileChangeHandler(), path='.', recursive=False)
    observer.start()
    print("Watching for new_order.json updates...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
