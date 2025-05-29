# python3 run.py --resume "./output/2025_5_9_exp1" --dataset fgvc
# python3 run.py --resume "./output/2025_5_9_exp2" --dataset fgvc,caltech101
# python3 run.py --resume "./output/2025_5_9_exp3" --dataset fgvc,caltech101,dtd
# python3 run.py --resume "./output/2025_5_9_exp4" --dataset fgvc,caltech101,dtd,eurosat
# python3 run.py --resume "./output/2025_5_9_exp5" --dataset fgvc,caltech101,dtd,eurosat,oxford_flowers
# python3 run.py --resume "./output/2025_5_9_exp6" --dataset fgvc,caltech101,dtd,eurosat,oxford_flowers,oxford_pets

import os

import matplotlib.pyplot as plt
import seaborn as sns

aircraft = [61.8, 61.7, 61.4, 61.3, 61.4, 61.1]
caltech101 = [None, 97.2, 97.2, 97.2, 97.3, 97.3]
dtd = [None, None, 79.3, 79.4, 79.3, 79.5]
eurosat = [None, None, None, 98.4, 98.4, 98.4]
flowers = [None, None, None, None, 97.9, 97.9]
oxfordpet = [None, None, None, None, None, 95.1]
datasets = ["aircraft", "caltech101", "dtd", "eurosat", "flowers", "oxfordpet"]
resume_dir = "./output/2025_5_9"
if __name__ == '__main__':

    plt.plot(datasets, aircraft, label="aircraft", marker='o')
    for x, y in zip(datasets, aircraft):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    plt.plot(datasets, caltech101, label="caltech101", marker='s')
    for x, y in zip(datasets, caltech101):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    plt.plot(datasets, dtd, label="dtd", marker='^')
    for x, y in zip(datasets, dtd):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    plt.plot(datasets, eurosat, label="eurosat", marker='v')
    for x, y in zip(datasets, eurosat):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    plt.plot(datasets, flowers, label="flowers", marker='D')
    for x, y in zip(datasets, flowers):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    plt.plot(datasets, oxfordpet, label="oxfordpet", marker='x')
    for x, y in zip(datasets, oxfordpet):
        if y is not None:
            plt.text(x, y + 0.4, f"{y}", ha='center', fontsize=8)
    sns.set_style("darkgrid")
    plt.title("Variation in accuracy")
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(resume_dir, "acc.png"), dpi=1000)
    plt.show()