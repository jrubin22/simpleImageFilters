import os
import subprocess
import csv
import time

# Define the executable and modes
EXECUTABLE = "../build/image_filter"
MODES = ["cpu", "gpu", "sobel-cpu", "sobel-gpu"]

# Directories
IMAGE_DIR = "../data/STI"
OUTPUT_DIR = "../output"
RESULTS_FILE = "../benchmarks/results/results.csv"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all image paths
image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.lower().endswith((".ppm", ".bmp", ".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))

# Run benchmarks
results = []
for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    for mode in MODES:
        out_img = os.path.join(OUTPUT_DIR, f"{base_name}-{mode}.jpg")
        cmd = [EXECUTABLE, img_path, out_img, mode]
        print(f"Running: {' '.join(cmd)}")

        try:
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            end = time.time()
            elapsed_ms = int((end - start) * 1000)

            # Parse kernel time from stdout
            kernel_time = None
            for line in result.stdout.splitlines():
                if "GPU kernel time" in line:
                    kernel_time = float(line.split(":")[-1].replace("ms", "").strip())

            results.append({
                "image": os.path.relpath(img_path, IMAGE_DIR),
                "mode": mode,
                "host_time_ms": elapsed_ms,
                "kernel_time_ms": kernel_time if kernel_time else "",
                "status": "success"
            })
        except subprocess.TimeoutExpired:
            results.append({
                "image": os.path.relpath(img_path, IMAGE_DIR),
                "mode": mode,
                "host_time_ms": "",
                "kernel_time_ms": "",
                "status": "timeout"
            })
        except Exception as e:
            results.append({
                "image": os.path.relpath(img_path, IMAGE_DIR),
                "mode": mode,
                "host_time_ms": "",
                "kernel_time_ms": "",
                "status": f"error: {e}"
            })

# Save results to CSV
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "mode", "host_time_ms", "kernel_time_ms", "status"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n Benchmark complete. Results saved to {RESULTS_FILE}")