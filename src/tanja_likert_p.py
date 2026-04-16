from scipy.stats import wilcoxon
import pandas as pd
from TPTBox import Location

# import Counter
from collections import Counter

p = "/DATA/NAS/ongoing_projects/tanja/poi_prediction/prediction_files-no_proj/pred_GT_rating_TL.ods"


df = pd.read_excel(p)

# df = df[df["Rating (GT)"] <= 3]


gt_l = df["Rating (GT)"].to_list()[:-4]
pr_l = df["Rating (prediction)"].to_list()[:-4]

print(gt_l)
print(pr_l)

stat, p_value = wilcoxon(gt_l, pr_l)

print(f"Wilcoxon signed-rank test statistic: {stat}")
print(f"P-value: {p_value}")


N = len(gt_l)

gt_c = Counter(gt_l)
pr_c = Counter(pr_l)

print("GT Rating Distribution:")
for rating, count in gt_c.items():
    percentage = (count / N) * 100
    print(f"Rating {rating}: {count} ({percentage:.2f}%)")

print("\nPrediction Rating Distribution:")
for rating, count in pr_c.items():
    percentage = (count / N) * 100
    print(f"Rating {rating}: {count} ({percentage:.2f}%)")

# give average and std
gt_mean = sum(gt_l) / N
gt_std = (sum((x - gt_mean) ** 2 for x in gt_l) / N) ** 0.5

pr_mean = sum(pr_l) / N
pr_std = (sum((x - pr_mean) ** 2 for x in pr_l) / N) ** 0.5

print(f"\nGT Rating Mean: {gt_mean:.2f}, Std: {gt_std:.2f}")
print(f"Prediction Rating Mean: {pr_mean:.2f}, Std: {pr_std:.2f}")
