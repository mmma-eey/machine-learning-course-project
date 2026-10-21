# -*- coding: utf-8 -*-
import csv
from collections import Counter

BASE = r"c:\Users\lenovo\.trae-cn\attachments\6aafd49b57e3fa8efcd0bbb2"
OUT = r"c:\Users\lenovo\Desktop\dataset-for-task2\clean_dataset\submission_arbitrated.csv"

def load(name):
    d = {}
    with open(BASE + "\\" + name, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["ID"]] = row["Category"]
    return d

def load_probs(name):
    d = {}
    with open(BASE + "\\" + name, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["ID"]] = (float(row["P_Black-grass"]), float(row["P_Loose Silky-bent"]))
    return d

cn384 = load("1f8b2e58-6bc6-4b19-9952-09d42407d5e8_49f1ad68-27ee-4dba-97e8-e0baf7acd2cb_submission_clean_384.csv")
cn320 = load("ccacb7f6-4203-47b8-bd87-6659b5d8a2d8_295790bc-c25a-4aae-94be-f9bc87dbe7ad_submission_convnext_small.csv")
pseudo = load("40579558-d6c0-4211-bb33-62f2985976fc_21a8cfeb-444b-474d-8796-701fa3b9426d_submission_pseudo.csv")
probs = load_probs("b4b3f03d-8ad3-4257-bad7-0304ae8e8e4f_bfbfea28-5e73-43b8-9555-e453607e2ecd_binary_test_probs.csv")

BG, LSB = "Black-grass", "Loose Silky-bent"
ids = sorted(cn384.keys())
print("test 样本数:", len(ids))
print("binary 概率覆盖:", len(probs))

# binary 在 test 上的判别分布
bin_bg = sum(1 for k in ids if probs.get(k, (0,0))[0] >= 0.5)
print(f"\nbinary 判定分布: BG={bin_bg}, LSB={len(ids)-bin_bg}")

# 置信度分布
conf = [abs(probs[k][0] - 0.5) for k in ids]
def bucket(c):
    return "高(>=0.4)" if c >= 0.4 else ("中(0.25~0.4)" if c >= 0.25 else "低(<0.25)")
cc = Counter(bucket(c) for c in conf)
print("binary 置信度分布(相对0.5的偏差):", dict(cc))

# 主模型 cn384 在 BG/LSB 上的样本, 与 binary 分歧
disagree = []
for k in ids:
    base = cn384[k]
    if base not in (BG, LSB):
        continue
    pb, pl = probs.get(k, (0.5, 0.5))
    binv = BG if pb >= pl else LSB
    if binv != base:
        disagree.append((k, base, binv, pb, pl))

print(f"\ncn384 判成 BG/LSB 的样本数: {sum(1 for k in ids if cn384[k] in (BG,LSB))}")
print(f"binary 与 cn384 分歧(BG/LSB内): {len(disagree)}")
pairs = Counter((b, v) for _, b, v, _, _ in disagree)
for (b, v), n in pairs.most_common():
    print(f"  cn384={b:20s} -> binary={v:20s} : {n}")

# 分歧明细(按置信度排序)
disagree.sort(key=lambda x: -abs(x[3]-0.5))
print("\n=== 分歧明细(按 binary 置信度降序) ===")
for k, base, binv, pb, pl in disagree:
    c = abs(pb - 0.5)
    print(f"  {k}: cn384={base:18s} binary={binv:18s} P_BG={pb:.4f} conf={c:.3f}")

# --- 生成仲裁提交(默认阈值: 只翻转高置信 binary 分歧) ---
THRESH = 0.4   # binary 相对0.5的偏差 >= THRESH 且分歧 才翻转
flip = 0
out_rows = []
for k in ids:
    base = cn384[k]
    if base in (BG, LSB):
        pb, pl = probs.get(k, (0.5, 0.5))
        binv = BG if pb >= pl else LSB
        if binv != base and abs(pb - 0.5) >= THRESH:
            base = binv
            flip += 1
    out_rows.append((k, base))
print(f"\n阈值 {THRESH} 翻转 {flip} 张")

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "Category"])
    for k, c in out_rows:
        w.writerow([k, c])
print("已写:", OUT)
print("仲裁后预测分布:", dict(Counter(c for _, c in out_rows)))