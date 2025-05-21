import subprocess
import time

scripts_to_run = [
    "retrieval.py",
    "matching.py",
    "ranking_updated.py",
]

record = []

for i in range(3):

    start = time.time()

    for script in scripts_to_run:
        result = subprocess.run(["python", script], check=True, capture_output=True, text=True)
        print(result.stdout)  # Print standard output

    end = time.time()

    record.append(end - start)

avg = 0
for t in record:
    print(f"{t : .2f}")
    avg += t

print(f"The avergae time taken is: {avg/3 : .2f}")