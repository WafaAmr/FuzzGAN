import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import json

# root_path = 'mnist/eval/final/HQ'
# labels = 'mnist/eval/final/HQ/HQ-labels-TN.txt'
root_path = 'mnist/eval/final/LQ'
labels = 'mnist/eval/final/LQ/LQ-labels-TN.txt'
classes = [[{"valid": 0, "invalid": 0, "target": 0, "unknown": 0}] for _ in range(10)]
model = {"valid": 0, "invalid": 0, "target": 0, "unknown": 0}

# Open the file
with open(labels) as file:
    # Read each line
    for line in file:
        # Print the line
        data = json.loads(line)
        seed_path, seed_name = os.path.split(data["image_path"])
        path, seed_idx = os.path.split(seed_path)
        _, seed_class = os.path.split(path)
        seed_class = int(seed_class)

        status = data["accepted"]
        if status:
            classes[seed_class][0]["valid"] += 1
            model["valid"] += 1
        elif not status:
            # print(data["label"])
            if data["label"] == "Unknown":
                classes[seed_class][0]["unknown"] += 1
                model["unknown"] += 1
            elif isinstance(int(data["label"]), int):
              path = os.path.join(root_path, str(seed_class), seed_idx)
              # target = [int(f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
              classes[seed_class][0]["invalid"] += 1
              model["invalid"] += 1
              print(os.path.join(root_path, data["image_path"][9:]))
              print(data["label"])
              classes[int(data["label"])][0]["target"] += 1
              # model["target"] += 1

# num_classes = 10

# # Plotting
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
# axes = axes.ravel()  # Flatten the array of axes

# for i in range(num_classes):
#     labels = list(classes[i][0].keys())
#     counts = list(classes[i][0].values())
#     print(labels, counts)
#     axes[i].grid(True, alpha=0.25)
#     axes[i].bar(range(len(counts)), counts, color='skyblue', zorder=3)
#     axes[i].set_title(f"Class {i}")
#     axes[i].set_xticks(range(len(counts)))
#     axes[i].set_xticklabels(labels)
#     axes[i].set_ylim(0, 100)  # setting a fixed y-axis limit for better comparison

# fig.tight_layout()
# plt.savefig(f'validity.png')
# plt.close()



print(model)
# print(classes)
mls = model.values()
for i in range(10):
    ls = classes[i][0].values()
    print(f'{i} & {list(ls)[0]} & {list(ls)[1]} & {list(ls)[3]} & 100 & {list(ls)[2]} \\\\')
print(f'0-9 & {list(mls)[0]} & {list(mls)[1]} & {list(mls)[3]} & 1000 & - \\\\')
# mnist/eval/final/HQ/0/1707
# mnist/eval/final/HQ/img/HQ/0/116190/