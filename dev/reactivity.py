"""Сценарий реактивности из ТЗ (а–г) и цикл «скачать → сбросить → загрузить».

После каждого шага: все четыре раздела сайта открываются без ошибок (AppTest),
а ключевые результаты (метрики кластеров, прямые RANSAC, корреляции, PLS) меняются.
Запуск из корня проекта:  python dev/reactivity.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402
import data as D  # noqa: E402

PAGES = ['📋 Данные', '🧩 Кластеризация', '📈 Корреляции', '✅ Валидация']
Y = A.SORPTION_MMOL


def fingerprint(ds):
    f = ds.analysis_frame('mmol/g')
    sub = f.dropna(subset=[Y, 'ε'])
    Xs = A.scale(sub[[Y, 'ε']].to_numpy(float))
    _, _, k = max(A.kmeans_candidates(Xs))
    lab, _ = A.fit_kmeans(Xs, k)
    met = A.clustering_metrics(Xs, 'K-Means', lab, '')
    out = {'n': len(f), 'kmeans_sil': round(met['Silhouette'], 6), 'kmeans_CH': round(met['Calinski-Harabasz'], 4)}
    for d in ('β', 'DN'):
        r = A.analyze_descriptor(f, d, Y, require_positive=True)
        out[f'{d}_lines'] = [(round(m['slope'], 4), len(m['inliers_idx'])) for m in r['models']]
    corr = A.correlation_table(f, ds.used_descriptors(), Y)
    out['corr_descriptors'] = list(corr['Descriptor'])
    out['pearson_eps'] = round(float(corr.set_index('Descriptor').loc['ε', 'Pearson_r']), 6)
    pls = A.pls_validation(f, ['ε', 'α', 'β', 'π*', 'DN'], Y)
    out['pls_q2'] = round(pls['q2'], 6) if not pls.get('error') else pls['error']
    return out


def run_pages(ds, label):
    at = AppTest.from_file('../app.py', default_timeout=900)
    at.run()
    at.session_state['ds'] = ds
    problems = []
    for p in PAGES:
        at.segmented_control(key='page').set_value(p).run()
        problems += [f'{p}: {e.value}' for e in at.exception]
    print(f'  [{label}] разделы сайта: ' + ('без ошибок' if not problems else '; '.join(problems)))
    return not problems


def diff(a, b):
    return {k: (a[k], b[k]) for k in a if a[k] != b[k]}


ds = D.Dataset(D.read_excel_raw())
ds.preset('notebook')
base = fingerprint(ds)
print('Исходно:', base)
ok = run_pages(ds, 'исходно')

print('(а) выключить все амины')
ids = ds.values.loc[ds.values[D.CLASS_COL] == 'Амины', D.ROW_ID].tolist()
ds.set_included(ids, False, 'класс «Амины»')
fa = fingerprint(ds)
print('  изменилось:', diff(base, fa))
ok &= run_pages(ds, 'а')

print('(б) изменить ε у одного вещества (Pyridine: 12.4 → 30)')
rid = ds.values.loc[ds.values[D.NAME] == 'Pyridine', D.ROW_ID].iloc[0]
ds.set_value(rid, 'ε', '30')
fb = fingerprint(ds)
print('  изменилось:', diff(fa, fb))
ok &= run_pages(ds, 'б')

print('(в) добавить новое вещество со всеми дескрипторами')
fields = {D.NAME: 'Test probe', D.SORPTION: '8,5', D.SMILES: 'CCO', D.CLASS_COL: 'Протонодонорные'}
vals = {'ε': 24.5, 'α': 0.86, 'β': 0.75, 'π*': 0.54, 'DN': 19.0, 'PA, kJ/mol': 776.4, 'σEB, Å²': 23,
        'σW, Å²': 23, 'TPSA 3D (Mordred CPSA), Å²': 30, 'TopoPSA 2D (RDKit, N/O), Å²': 20.23,
        'Polar SASA 3D (Shrake–Rupley, N/O + H), Å²': 40}
fields.update({k: str(v).replace('.', ',') for k, v in vals.items()})
ds.add_substance(fields)
fc = fingerprint(ds)
print('  изменилось:', diff(fb, fc))
ok &= run_pages(ds, 'в')

print('(г) добавить новый дескриптор и заполнить его для 10 веществ')
ds.add_descriptor('Test descriptor', 'у.е.', 'проверочный', 'да')
f = ds.analysis_frame()
names = f[A.NAME].tolist()[:10]
ds.paste_column('Test descriptor', '\n'.join(f'{n}\t{v:.3f}'.replace('.', ',') for n, v in
                                             zip(names, f.set_index(A.NAME).loc[names, Y] * 0.3 + 1)),
                mode='pairs')
fd = fingerprint(ds)
print('  изменилось:', diff(fc, fd))
r = A.analyze_descriptor(ds.analysis_frame(), 'Test descriptor', Y, require_positive=True)
print(f'  RANSAC по новому дескриптору: {len(r["data"])} веществ, прямых {len(r["models"])}; '
      f'в списке анализа: {"Test descriptor" in ds.used_descriptors()}')
if r['models']:
    pt = A.permutation_test(ds.analysis_frame(), 'Test descriptor', r['models'], ycol=Y, n_permutations=100)
    print(f'  перестановочный тест по новому дескриптору: p = {pt["p"]:.4f}')
ok &= run_pages(ds, 'г')

print('Скачать → сбросить → загрузить')
blob = ds.to_xlsx()
reset = D.Dataset(D.read_excel_raw())
restored = D.Dataset.from_xlsx(io.BytesIO(blob))
same_values = all(ds.values[c].astype(object).equals(restored.values[c].astype(object)) for c in ds.values.columns)
same = (same_values and list(ds.values.columns) == list(restored.values.columns)
        and ds.descriptors == restored.descriptors and ds.meta == restored.meta
        and len(ds.journal) == len(restored.journal))
print(f'  после сброса веществ {reset.summary()[1]}, после загрузки {restored.summary()[1]}; '
      f'состояние восстановлено полностью: {same}; отпечаток совпал: {fingerprint(restored) == fd}')
ok &= same
print('ИТОГ:', 'всё в порядке' if ok else 'есть проблемы')
