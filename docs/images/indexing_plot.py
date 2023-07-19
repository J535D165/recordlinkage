import matplotlib.colors as mlc
import matplotlib.pyplot as mlp
import numpy as np

figure, axes = mlp.subplots(nrows=1, ncols=2, figsize=(8, 5))

# linking
db_a = ["A1", "A2", "A3", "A4", "A5", "A6"]
db_b = ["B1", "B2", "B3", "B4", "B5", "B6"]

img = np.ones((len(db_a), len(db_b)), dtype=float)

color_map = mlc.LinearSegmentedColormap.from_list(
    "ColorMap", [(0.984, 0.501, 0.447), (1.000, 1.000, 1.000)]
)
axes[0].imshow(img, cmap=color_map, interpolation="none")

axes[0].set_xlabel("Dataset A", fontsize=13)
axes[0].set_xticks(np.arange(0, len(db_b), 1))
axes[0].set_xticks(np.arange(-0.5, len(db_b), 1), minor=True)
axes[0].set_xticklabels(db_a)

axes[0].set_ylabel("Dataset B", fontsize=13)
axes[0].set_yticks(np.arange(0, len(db_a), 1))
axes[0].set_yticks(np.arange(-0.5, len(db_a), 1), minor=True)
axes[0].set_yticklabels(db_b)

axes[0].grid(which="minor", color="k")

axes[0].set_title("Linking A and B", fontsize=15, fontweight="bold")

# dedup
db_a = ["A1", "A2", "A3", "A4", "A5", "A6"]
db_b = ["A1", "A2", "A3", "A4", "A5", "A6"]

img = np.ones((len(db_a), len(db_b)), dtype=float)
img = np.triu(img, 1)

color_map = mlc.LinearSegmentedColormap.from_list(
    "ColorMap", [(1.000, 1.000, 1.000), (0.984, 0.501, 0.447)]
)
axes[1].imshow(img, cmap=color_map, interpolation="none")

axes[1].set_xlabel("Dataset A", fontsize=13)
axes[1].set_xticks(np.arange(0, len(db_b), 1))
axes[1].set_xticks(np.arange(-0.5, len(db_b), 1), minor=True)
axes[1].set_xticklabels(db_a)

axes[1].set_ylabel("Dataset A", fontsize=13)
axes[1].set_yticks(np.arange(0, len(db_a), 1))
axes[1].set_yticks(np.arange(-0.5, len(db_a), 1), minor=True)
axes[1].set_yticklabels(db_b)

axes[1].grid(which="minor", color="k")

axes[1].set_title("Duplicate detection A", fontsize=15, fontweight="bold")

figure.tight_layout()

mlp.savefig("indexing_basic.png", dpi=150)
