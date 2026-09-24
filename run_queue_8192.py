import queue
import subprocess
import threading

# endpoint -> allowed parallelism
ENDPOINTS = [
    ("http://192.168.2.2:8080/v1", 1),
    ("http://192.168.2.2:8081/v1", 1),
    ("http://192.168.2.2:8082/v1", 1),
    ("http://192.168.2.2:8083/v1", 1),
]

COMMON_CMD = [
    "python",
    "run.py",
    "--reasoning-effort",
    "low",
    "--n_samples",
    "1",
    "--seed",
    "857",
    "--note",
    "rb8192",
]

q = queue.Queue()

# for printing
lock = threading.Lock()

def log(msg):
    with lock:
        print(msg, flush=True)

def worker(base_url):
    while True:
        try:
            n_skip = q.get_nowait()
        except queue.Empty:
            return

        cmd = [*COMMON_CMD, "--base-url", base_url, "--n_skip", str(n_skip)]

        log(f"I: {n_skip} on {base_url}: start")
        #print(" ".join(cmd))
        #res = 0
        res = subprocess.run(cmd).returncode
        
        if res != 0:
            log(f"E: {n_skip} on {base_url}: error {res}")
        else:
            log(f"I: {n_skip} on {base_url}: done")

def main():
    n_skip_list = list(range(0, 40))
    for n_skip in n_skip_list:
        q.put(n_skip)

    # allocate slots
    slots = []
    max_parallelism = max(p for _, p in ENDPOINTS)
    for i in range(max_parallelism):
        for base_url, p in ENDPOINTS:
            if i < p:
                slots.append(base_url)

    threads = [threading.Thread(target=worker, args=(base_url,)) for base_url in slots]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()