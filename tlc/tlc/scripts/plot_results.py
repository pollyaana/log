#!/usr/bin/env python3
"""Графики по results/kernels.csv: время и пик памяти от n (лог-шкала)."""
import csv, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'results/kernels.csv'
out = os.path.dirname(src) or '.'
rows = list(csv.DictReader(open(src)))
impls = {'analog2_naive': ('Аналог 2: степени матриц, O(n^4)', '#c0504d'), 'tlc_fast': ('tlc: Карп + Флойд–Уоршелл, O(n^3)', '#2e6fb5')}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for key, (label, color) in impls.items():
    r = [x for x in rows if x['impl'] == key]
    n = [int(x['n']) for x in r]
    ax[0].plot(n, [float(x['median_ms']) for x in r], 'o-', color=color, label=label)
    ax[1].plot(n, [int(x['peak_heap_bytes']) / 1024 for x in r], 'o-', color=color, label=label)
    bad = [int(x['n']) for x in r if x['finite'] == '0']
    if bad:
        ax[0].axvspan(min(bad) - 5, max(n) + 5, color=color, alpha=0.08)
        ax[0].text(min(bad), ax[0].get_ylim()[1] if False else float(r[-1]['median_ms']) / 30, ' λ = inf\n (переполнение)', color=color)
for a, t, y in [(ax[0], 'Время λ + звезда Клини (медиана)', 'мс'), (ax[1], 'Пик памяти кучи', 'КиБ')]:
    a.set_xlabel('n'); a.set_ylabel(y); a.set_title(t); a.set_yscale('log'); a.grid(alpha=0.3); a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, 'kernels.png'), dpi=150)
print('saved', os.path.join(out, 'kernels.png'))
