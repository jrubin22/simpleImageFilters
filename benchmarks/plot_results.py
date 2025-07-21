import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load benchmark data
df = pd.read_csv("results/results.csv")

# Filter only the relevant 4 modes
df = df[df["mode"].isin(["cpu", "gpu", "sobel-cpu", "sobel-gpu"])]
df = df[df["status"] == "success"]
df["image"] = df["image"].apply(lambda x: x.split("/")[-1])

# Convert times to numeric
df["host_time_ms"] = pd.to_numeric(df["host_time_ms"], errors="coerce")
df["kernel_time_ms"] = pd.to_numeric(df["kernel_time_ms"], errors="coerce")

sns.set(style="whitegrid")

# Grayscale comparison (gpu vs cpu)
gray_df = df[df["mode"].isin(["cpu", "gpu"])]
plt.figure(figsize=(20, 6))
sns.barplot(data=gray_df, x="image", y="host_time_ms", hue="mode")
plt.title("CPU vs GPU Grayscale Filter: Host Time")
plt.ylabel("Time (ms)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("grayscale_host_comparison.png")
print("Saved: grayscale_host_comparison.png")

# Sobel comparison (sobel-cpu vs sobel-gpu)
sobel_df = df[df["mode"].isin(["sobel-cpu", "sobel-gpu"])]
plt.figure(figsize=(10, 5))
sns.barplot(data=sobel_df, x="image", y="host_time_ms", hue="mode")
plt.title("CPU vs GPU Sobel Filter: Host Time")
plt.ylabel("Time (ms)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("sobel_host_comparison.png")
print("Saved: sobel_host_comparison.png")

# Optional: GPU-only kernel time comparison
gpu_only = df[df["mode"].isin(["gpu", "sobel-gpu"])].dropna(subset=["kernel_time_ms"])
if not gpu_only.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=gpu_only, x="image", y="kernel_time_ms", hue="mode")
    plt.title("GPU Kernel Time (Grayscale vs Sobel)")
    plt.ylabel("Kernel Time (ms)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("gpu_kernel_time_comparison.png")
    print("Saved: gpu_kernel_time_comparison.png")
