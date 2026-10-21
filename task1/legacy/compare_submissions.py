# -*- coding: utf-8 -*-
import csv
from collections import Counter, defaultdict

PSEUDO = r"c:\Users\lenovo\.trae-cn\attachments\6aafd49b57e3fa8efcd0bbb2\40579558-d6c0-4211-bb33-62f2985976fc_21a8cfeb-444b-474d-8796-701fa3b9426d_submission_pseudo.csv"
CONVNEXT = r"c:\Users\lenovo\.trae-cn\attachments\6aafd49b57e3fa8efcd0bbb2\ccacb7f6-4203-47b8-bd87-6659b5d8a2d8_295790bc-c25a-4aae-94be-f9bc87dbe7ad_submission_convnext_small.csv"

def load(path):
    m = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m[row["ID"]] = row["Category"]
    return m

a = load(PSEUDO)
b = load(CONVNEXT)

print("pseudo 条数:", len(a))
print("convnext 条数:", len(b))
print("ID 集合是否一致:", set(a) == set(b))

diff = []
for k in a:
    if a[k] != b[k]:
        diff.append((k, a[k], b[k]))

print("分歧样本数:", len(diff))
print()
print("分歧明细 (ID | pseudo | convnext):")
for d in diff:
    print(f"  {d[0]}  {d[1]}  ->  {d[2]}")

# 混淆转移统计
trans = Counter((x[1], x[2]) for x in diff)
print()
print("分歧转移对 (pseudo -> convnext) 计数:")
for (p, c), n in sorted(trans.items(), key=lambda kv: -kv[1]):
    print(f"  {p} -> {c}: {n}")