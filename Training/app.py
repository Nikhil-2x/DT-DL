import os
import time

job = os.getenv("JOB_ID", "unknown")
partition = os.getenv("PARTITION", "0")

print(f"Job ID: {job}")
print(f"Training partition: {partition}")

print("Loading dataset...")
time.sleep(2)

for epoch in range(5):
    print(f"Partition {partition} - Epoch {epoch+1}/5")
    time.sleep(1)

print(f"Partition {partition} finished")
