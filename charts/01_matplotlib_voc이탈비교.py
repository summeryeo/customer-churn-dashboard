"""전체 고객 이탈율 vs 해지관련 부정 VOC 이력 고객 이탈율 비교 막대그래프"""
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"))
voc = pd.read_csv(os.path.join(DATA_DIR, "data_voc.csv"))

# 전체 고객 이탈율
total_customers = len(customers)
total_churn = (customers["churn_yn"] == "Y").sum()
total_churn_rate = total_churn / total_customers * 100

# 해지관련 + 부정 VOC를 남긴 고객 목록 (중복 제거)
target_ids = voc.loc[
    (voc["category"] == "해지관련") & (voc["sentiment"] == "부정"), "customer_id"
].unique()

# customer_id로 연결하여 이탈율 계산
target_customers = customers[customers["customer_id"].isin(target_ids)]
target_total = len(target_customers)
target_churn = (target_customers["churn_yn"] == "Y").sum()
target_churn_rate = target_churn / target_total * 100

labels = ["전체 고객", "해지관련 부정 VOC 이력 있음"]
rates = [total_churn_rate, target_churn_rate]
colors = ["#2a78d6", "#d03b3b"]  # 기본: 파랑, 강조: 빨강

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, rates, color=colors, width=0.5)

for bar, rate in zip(bars, rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

ax.set_ylabel("이탈율 (%)")
ax.set_title("전체 고객 vs 해지관련 부정 VOC 고객 이탈율 비교")
ax.set_ylim(0, max(rates) * 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "01_matplotlib_voc이탈비교.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"saved: {output_path}")
