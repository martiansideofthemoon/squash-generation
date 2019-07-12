import time

key = None

while True:
    time.sleep(0.1)
    with open("squash/temp/queue.txt", "r") as f:
        data = f.read().strip()
    if len(data) == 0:
        continue
    else:
        key = data.split("\n")[0]
        print(key)
        break
