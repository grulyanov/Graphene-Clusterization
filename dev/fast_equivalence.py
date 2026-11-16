"""Проверка, что быстрый движок (analysis.USE_FAST = True) даёт те же результаты, что
исходные sklearn-реализации ноутбука, и замер ускорения.

Запуск из корня проекта:  python dev/fast_equivalence.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402
import data as D  # noqa: E402

rng = np.random.default_rng(0)
n_cases = mism_mask = mism_err = 0
max_coef = 0.0
t_fast = t_ref = 0.0
for case in range(1500):
    n = int(rng.integers(6, 40))
    p = 1 if case % 5 else 2
    X = np.round(rng.normal(size=(n, p)) * rng.choice([1, 10, 100]), int(rng.integers(0, 3)))
    y = np.round(X @ rng.normal(size=p) + rng.standard_t(2, size=n) * rng.choice([0.2, 1, 3]), 3)
    tau = float(rng.choice(A.TAU_GRID_MMOL_G))
    ms = int(rng.integers(p + 1, min(p + 4, n)))
    seed = int(rng.integers(0, 2 ** 32 - 1)) if case % 3 == 0 else 42
    mt = int(rng.choice([300, 2000, 20000]))
    out = []
    for fast in (True, False):
        A.USE_FAST = fast
        t0 = time.perf_counter()
        try:
            m = A.ransac_fit(X, y, ms, tau, mt, 0.999, seed)
            out.append((m.inlier_mask_.copy(), np.r_[np.ravel(m.estimator_.coef_), m.estimator_.intercept_]))
        except ValueError:
            out.append(None)
        if fast:
            t_fast += time.perf_counter() - t0
        else:
            t_ref += time.perf_counter() - t0
    n_cases += 1
    if (out[0] is None) != (out[1] is None):
        mism_err += 1
    elif out[0] is not None:
        if not np.array_equal(out[0][0], out[1][0]):
            mism_mask += 1
        else:
            max_coef = max(max_coef, float(np.max(np.abs(out[0][1] - out[1][1]))))
A.USE_FAST = True
print(f'RANSAC: {n_cases} случайных наборов; маски inliers различаются: {mism_mask}; '
      f'исключения различаются: {mism_err}; макс. разница коэффициентов: {max_coef:.2e}; '
      f'ускорение ×{t_ref / t_fast:.0f}')

# LOO Q²
diffs = []
for case in range(500):
    n = int(rng.integers(4, 30))
    x = rng.normal(size=n)
    if case % 10 == 0:
        x[:] = x[0]
    y = 2 * x + rng.normal(size=n)
    a = A.loo_q2(x, y)
    b = A._loo_q2_reference(x, y)
    diffs.append(0.0 if (a == b or (np.isinf(a) and np.isinf(b))) else abs(a - b))
print(f'LOO Q²: 500 наборов, макс. разница {max(diffs):.2e}')

# Время на реальных данных
df = D.read_excel_raw()
df = df.loc[~A.notebook_excluded_mask(df['Name'])].reset_index(drop=True)
descs = ['ε', 'α', 'β', 'π*', 'DN', 'PA, kJ/mol', 'σEB, Å²', 'σW, Å²', 'TPSA 3D (Mordred CPSA), Å²',
         'TopoPSA 2D (RDKit, N/O), Å²', 'Polar SASA 3D (Shrake–Rupley, N/O + H), Å²']
for fast in (True, False):
    A.USE_FAST = fast
    A._SUBSETS = None
    t0 = time.perf_counter()
    for d in descs:
        A.analyze_descriptor(df, d, require_positive=d in ('α', 'β', 'π*', 'DN'))
    print(f'RANSAC по 11 дескрипторам, 1 ядро, {"быстрый" if fast else "исходный"} движок: '
          f'{time.perf_counter() - t0:.1f} c')
A.USE_FAST = True
