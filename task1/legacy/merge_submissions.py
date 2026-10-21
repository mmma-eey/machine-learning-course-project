# -*- coding: utf-8 -*-
"""
合并两个提交文件: 按 ID 对齐投票, 分歧时采信更高分的模型.
用法: python merge_submissions.py <csvA> <csvB> <输出csv> [--tie 文件名前缀]
说明: 两个文件均需为 ID,Category 格式. 默认分歧时:(A与B都写, 需指定哪个更可信).
"""
import sys
import csv
from collections import defaultdict


def load(path):
    d = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) == 2:
                d[row[0]] = row[1].strip()
    return d


def merge(a, b, tie_winner="A"):
    ids = sorted(set(a) | set(b), key=str)
    out = {}
    for i in ids:
        if i in a and i in b:
            va, vb = a[i], b[i]
            out[i] = va if va == vb else (a[i] if tie_winner == "A" else b[i])
        else:
            out[i] = a.get(i, b.get(i))
    return ids, out


def main():
    if len(sys.argv) < 4:
        print("用法: python merge_submissions.py <A.csv> <B.csv> <out.csv> [--tie A|B]")
        return
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    tie = "A"
    if len(sys.argv) >= 5 and sys.argv[4] == "--tie":
        tie = sys.argv[5]
    ids, out = merge(a, b, tie)
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Category"])
        for i in ids:
            w.writerow([i, out[i]])
    from collections import Counter
    diff = sum(1 for k in ids if k in a and k in b and a[k] != b[k])
    print(f"A={len(a)}行  B={len(b)}行  合并={len(ids)}行  两者分歧={diff}张 (采用tie={'A' if tie=='A' else 'B'})")
    print("合并后各类分布:", dict(Counter(out.values())))


if __name__ == "__main__":
    main()