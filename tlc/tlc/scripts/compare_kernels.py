#!/usr/bin/env python3
"""Сверка вывода `tlc kernels` и `analog2_cli kernels` (одинаковый формат).
usage: compare_kernels.py out_tlc.txt out_analog.txt [rel_tol=1e-9]"""
import re, sys

def parse(path):
    blocks, cur = {}, None
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\s*(\S+): lambda = (\S+)', line)
        if m:
            cur = m.group(1); blocks[cur] = {'lambda': float(m.group(2)), 'star': []}
        elif cur and line.strip().startswith('['):
            blocks[cur]['star'].append([float(v) for v in line.strip(' []\n').split()])
    return blocks

def rel(a, b): return abs(a - b) / max(1.0, abs(a), abs(b))

a, b = parse(sys.argv[1]), parse(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
bad = 0
for name in a:
    if name not in b: print(f'{name}: missing in {sys.argv[2]}'); bad += 1; continue
    dl = rel(a[name]['lambda'], b[name]['lambda'])
    ds = max((rel(x, y) for ra, rb in zip(a[name]['star'], b[name]['star']) for x, y in zip(ra, rb)), default=0)
    ok = dl <= tol and ds <= max(tol, 1e-6)   # звезда печатается с 6 знаками
    bad += not ok
    print(f'{name:4s} lambda_rel_diff={dl:.2e} star_max_rel_diff={ds:.2e} {"OK" if ok else "MISMATCH"}')
sys.exit(1 if bad else 0)
