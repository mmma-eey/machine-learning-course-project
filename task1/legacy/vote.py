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

cn384 = load("1f8b2e58-6bc6-4b19-9952-09d42407d5e8_49f1ad68-27ee-4dba-97e8-e0baf7acd2cb_submission_clean_384.csv")
cn320 = load("ccacb7f6-4203-47b8-bd87-6659b5d8a2d8_295790bc-c25a-4aae-94be-f9bc87dbe7ad_submission_convnext_small.csv")
pseudo = load("40579558-d6c0-4211-bb33-62f2985976fc_21a8cfeb-444b-474d-8796-701fa3b9426d_submission_pseudo.csv")

ids = sorted(cn384.keys())

# 各模型 BG/LSB 分布
for name, d in [("pseudo", pseudo), ("cn320", cn320), ("cn384", cn384)]:
    c = Counter(d[k] for k in ids)
    print(f"{name}: {dict(c)}")

# 多数投票(3模型)
def vote(k):
    c = Counter([pseudo[k], cn320[k], cn384[k]])
    return c.most_common(1)[0][0]

vote_pred = {k: vote(k) for k in ids}
print("\n多数投票分布:", dict(Counter(vote_pred.values())))

# 投票 vs cn384 差异
diff = [(k, cn384[k], vote_pred[k]) for k in ids if cn384[k] != vote_pred[k]]
print(f"多数投票与 cn384 不一致: {len(diff)} 张")
print("差异方向:", dict(Counter((a, b) for _, a, b in diff)))