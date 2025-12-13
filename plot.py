import re
import matplotlib.pyplot as plt

LOG_PATH = "logs.txt"  # change if needed

# Storage
frames = {"water": [], "no_water": []}
true_values = {"water": [], "no_water": []}
pred_values = {"water": [], "no_water": []}

# Regular expression to parse log lines
line_re = re.compile(
    r".*/(water|no_water)/frame_(\d+)\.jpg: true (\d+), predicted (\d+)"
)

# --- Parse logs ---
with open(LOG_PATH, "r") as f:
    for line in f:
        m = line_re.search(line.strip())
        if not m:
            continue

        cls = m.group(1)
        frame = int(m.group(2))
        true_label = int(m.group(3))
        pred_label = int(m.group(4))

        frames[cls].append(frame)
        true_values[cls].append(true_label)
        pred_values[cls].append(pred_label)


# --- Make a plot for each class ---
for cls in ["water", "no_water"]:
    if not frames[cls]:
        continue

    # Sort by frame order
    sorted_idx = sorted(range(len(frames[cls])), key=lambda i: frames[cls][i])

    x = [frames[cls][i] for i in sorted_idx]
    y_true = [true_values[cls][i] for i in sorted_idx]
    y_pred = [pred_values[cls][i] for i in sorted_idx]

    plt.figure(figsize=(10, 4))

    plt.plot(x, y_true, label="True", linewidth=2)
    plt.plot(x, y_pred, label="Predicted", linewidth=1.5)

    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1])
    plt.xlabel("Frame number (time)")
    plt.ylabel("Label")
    plt.title(f"{cls} — True vs Predicted Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
