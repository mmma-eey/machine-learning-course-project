# -*- coding: utf-8 -*-
import csv
from collections import Counter

BASE = r"c:\Users\lenovo\.trae-cn\attachments\6aafd49b57e3fa8efcd0bbb2"

def load(name):
    d = {}
    with open(BASE + "\\" + name, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["ID"]] = row["Category"]
    return d

pseudo = load("40579558-d6c0-4211-bb33-62f2985976fc_21a8cfeb-444b-474d-8796-701fa3b9426d_submission_pseudo.csv")
cn320  = load("ccacb7f6-4203-47b8-bd87-6659b5d8a2d8_295790bc-c25a-4aae-94be-f9bc87dbe7ad_submission_convnext_small.csv")
cn384  = load("1f8b2e58-6bc6-4b19-9952-09d42407d5e8_49f1ad68-27ee-4dba-97e8-e0baf7acd2cb_submission_clean_384.csv")

ids = sorted(cn384.keys())
print("总数:", len(ids))

def cmp(nameA, A, nameB, B):
    diff = [(k, A[k], B[k]) for k in ids if A.get(k) != B.get(k)]
    print(f"\n=== {nameA} vs {nameB}: 分歧 {len(diff)} 张 ===")
    pairs = Counter((a, b) for _, a, b in diff)
    for (a, b), c in pairs.most_common():
        print(f"  {a:20s} -> {b:20s} : {c}")
    return diff

cmp("pseudo", pseudo, "cn320", cn320)
cmp("cn320", cn320, "cn384", cn384)
cmp("pseudo", pseudo, "cn384", cn384)

# 三方对比
d3 = {}
for k in ids:
    d3[k] = (pseudo.get(k), cn320.get(k), cn384.get(k))

print("\n=== 三方完全一致 ===", sum(1 for v in d3.values() if len(set(v)) == 1))
print("=== 三方两两有分歧的样本 ===")
for k in ids:
    v = d3[k]
    if len(set(v)) > 1:
        print(f"  {k}: pseudo={v[0]}  cn320={v[1]}  cn384={v[2]}")