"""Интерактивный сайт «Сорбция молекулярных зондов на оксиде графена».

Запуск:  streamlit run app.py
Все расчёты берут данные только из st.session_state['ds'] (data.Dataset).
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler, StandardScaler

import analysis as A
import data as D
import plots as P
from explain import T

st.set_page_config(page_title='Сорбция зондов на оксиде графена', page_icon='🧪', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.headline {background: rgba(49,51,63,0.06); border-radius: 8px; padding: 0.55rem 0.9rem;
           margin-bottom: 0.8rem; font-size: 0.95rem;}
.small-note {color: #6b6f7b; font-size: 0.86rem;}
div[data-testid="stExpander"] summary p {font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

N_CPU = os.cpu_count() or 2
PAGES = ['📋 Данные', '🧩 Кластеризация', '📈 Корреляции', '✅ Валидация']


# =============================================================================
# Состояние
# =============================================================================
@st.cache_data(show_spinner=False)
def _raw_excel():
    return D.read_excel_raw()


def new_dataset():
    ds = D.Dataset(_raw_excel())
    ds.preset('notebook')          # по умолчанию — как в ноутбуке filtered
    ds.journal.clear()
    ds.undo_stack.clear()
    ds._log('Начальный набор: как в ноутбуке filtered',
            'исключены n-бутиламин, морфолин, пиперидин, PBr3, SOCl2')
    return ds


if 'ds' not in st.session_state:
    st.session_state.ds = new_dataset()
    st.session_state.editor_v = 0
    st.session_state.meta_v = 0
    st.session_state.messages = []
ds: D.Dataset = st.session_state.ds


def flash(kind, text):
    st.session_state.messages.append((kind, text))


def show_flash():
    for kind, text in st.session_state.messages:
        getattr(st, kind)(text)
    st.session_state.messages = []


# =============================================================================
# Помощники вывода
# =============================================================================
def explain(key, title='Что это значит?'):
    with st.expander(f'ℹ️ {title}'):
        st.markdown(T[key])


def _csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def _xlsx_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='table')
    return buf.getvalue()


def _safe(name):
    keep = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in name)
    return keep.strip('_')[:80] or 'table'


def show_table(df: pd.DataFrame, name: str, key: str, height='auto', round_to=4, **kwargs):
    """Таблица с копированием (выделите ячейки и Ctrl/Cmd+C) и скачиванием CSV / Excel."""
    if df is None or len(df) == 0:
        st.caption('— таблица пуста —')
        return
    shown = df.copy()
    if round_to is not None:
        num = shown.select_dtypes(include='number').columns
        shown[num] = shown[num].round(round_to)
    st.dataframe(shown, hide_index=True, height=height, width='stretch', **kwargs)
    c1 = c2 = st.container(horizontal=True)
    c1.download_button('⬇ CSV', data=lambda d=df: _csv_bytes(d), file_name=f'{_safe(name)}.csv',
                       mime='text/csv', key=f'csv_{key}', on_click='ignore')
    c2.download_button('⬇ Excel', data=lambda d=df: _xlsx_bytes(d), file_name=f'{_safe(name)}.xlsx',
                       key=f'xlsx_{key}', on_click='ignore')


def show_fig(fig, name: str, key: str, max_width=1300):
    png = P.to_png(fig, dpi=110)
    w, _ = fig.get_size_inches()
    st.image(png, width=int(min(max_width, w * 95)))
    c1 = c2 = st.container(horizontal=True)
    c1.download_button('⬇ PNG 600 dpi', data=lambda f=fig: P.to_png(f, 600), file_name=f'{_safe(name)}.png',
                       mime='image/png', key=f'png_{key}', on_click='ignore')
    c2.download_button('⬇ SVG', data=lambda f=fig: P.to_svg(f), file_name=f'{_safe(name)}.svg',
                       mime='image/svg+xml', key=f'svg_{key}', on_click='ignore')


def headline():
    n_in, n_all, off = ds.summary()
    off_txt = ', '.join(off) if off else 'нет'
    unit = D.UNITS[st.session_state.get('units', 'mmol/g')]['title']
    st.markdown(f'<div class="headline">В расчёте <b>{n_in}</b> веществ из <b>{n_all}</b>; '
                f'выключены: {off_txt}. Сорбция в единицах: <b>{unit}</b>.</div>',
                unsafe_allow_html=True)


def used_note(n_used, missing, what='в расчёт'):
    txt = f'Вошло {what}: **{n_used}** веществ.'
    if missing:
        txt += f' Не вошли из-за пустых значений ({len(missing)}): ' + ', '.join(missing) + '.'
    st.caption(txt)


def fhash(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
            h.update(repr(list(o.columns)).encode())
        elif isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()


# =============================================================================
# Хранилище тяжёлых RANSAC-расчётов: считается только то, чего ещё нет, параллельно
# =============================================================================
@st.cache_resource
def _store():
    return OrderedDict()


def compute_many(tasks: dict) -> dict:
    """tasks: ключ → (функция, args, kwargs). Возвращает ключ → результат."""
    store = _store()
    missing = [k for k in tasks if k not in store]
    if missing:
        with st.spinner(f'Считаю RANSAC ({len(missing)} расчёт(а/ов))…'):
            if len(missing) == 1:
                f, a, kw = tasks[missing[0]]
                results = [f(*a, **kw)]
            else:
                results = Parallel(n_jobs=min(len(missing), N_CPU))(
                    delayed(tasks[k][0])(*tasks[k][1], **tasks[k][2]) for k in missing)
        for k, r in zip(missing, results):
            store[k] = r
        while len(store) > 400:
            store.popitem(last=False)
    return {k: store[k] for k in tasks}


# =============================================================================
# Боковая панель: единицы, зерно, RANSAC, список веществ
# =============================================================================
def units_key():
    return st.session_state.get('units', 'mmol/g')


def default_grid(units):
    f = D.UNITS[units]['tau_factor']
    return ', '.join(A.format_tau_units(v * f, f) for v in A.TAU_GRID_MMOL_G)


def parse_grid(text):
    vals = []
    for part in str(text).replace(';', ' ').split():
        part = part.strip().strip(',')
        if part:
            vals.append(D.parse_number(part))
    if not vals and text.strip():
        vals = [D.parse_number(p) for p in text.split(';')]
    vals = sorted({v for v in vals if np.isfinite(v) and v > 0})
    if not vals:
        raise ValueError('Сетка τ пуста.')
    return tuple(vals)


def _toggle_one(rid):
    ds.set_included([rid], bool(st.session_state[f'inc_{rid}']), 'боковая панель')
    st.session_state.editor_v += 1


def sidebar():
    sb = st.sidebar
    sb.markdown('## ⚙️ Настройки анализа')
    avail = ds.units_available()
    options = [u for u in D.UNITS if avail[u]]
    if units_key() not in options:
        st.session_state.units = 'mmol/g'
    sb.radio('Единицы сорбции', options, key='units', format_func=lambda u: D.UNITS[u]['title'])
    if not avail['qVm']:
        sb.caption('q·Vm появится, когда у всех включённых веществ будет заполнена плотность '
                   '(столбец «Плотность, г/см³» на вкладке «Данные»).')
    if not avail['g/g']:
        sb.caption('г/г недоступно: у части включённых веществ нет молярной массы M.')
    with sb.expander('ℹ️ Что это значит?'):
        st.markdown(T['units'])
    sb.number_input('Зерно случайности (seed)', min_value=0, max_value=2 ** 31 - 1, value=42, step=1,
                    key='seed')
    with sb.expander('ℹ️ Зачем зерно?'):
        st.markdown(T['seed'])

    with sb.expander('🎯 Настройки RANSAC', expanded=False):
        st.radio('Порог τ', ['Автоматически (сетка, как в ноутбуке)', 'Фиксированный'], key='tau_mode')
        u = units_key()
        st.number_input(f'Фиксированный τ ({D.UNITS[u]["short"]})', min_value=0.0001,
                        value=1.0 * D.UNITS[u]['tau_factor'], step=0.05 * D.UNITS[u]['tau_factor'],
                        format='%.4f', key=f'tau_fixed_{u}',
                        disabled=st.session_state.get('tau_mode', '').startswith('Авто'))
        st.text_input(f'Сетка τ ({D.UNITS[u]["short"]})', value=default_grid(u), key=f'tau_grid_{u}',
                      help=T['tau_grid'])
        st.caption(T['tau'])
        st.slider('Минимальная доля точек на прямой, %', 5, 60, 20, key='min_fraction', help=T['min_inliers'])
        st.number_input('…но не меньше, точек', 2, 20, 4, key='min_floor')
        st.slider('Минимальный R²_in', 0.0, 0.99, 0.50, 0.01, key='min_r2', help=T['min_r2'])
        st.slider('Минимальный Q²_LOO', -1.0, 0.99, 0.50, 0.01, key='min_q2', help=T['min_q2'])
        st.slider('Максимум прямых', 1, 9, 5, key='max_lines', help=T['max_lines'])
        st.select_slider('max_trials', [500, 1000, 2000, 5000, 10000, 20000, 50000], 20000,
                         key='max_trials', help=T['max_trials'])
        st.radio('Положительный наклон', ['По свойству дескриптора', 'Требовать всегда', 'Не требовать'],
                 key='positive_mode', help=T['positive'])
        st.checkbox('Доверительная полоса у прямых', True, key='ci_on', help=T['ci'])
        st.select_slider('Уровень доверия', [0.90, 0.95, 0.99], 0.95, key='ci_level',
                         format_func=lambda v: f'{int(v * 100)} %', disabled=not st.session_state.get('ci_on', True))

    base_default = [d for d in ['ε', 'α', 'β', 'π*', 'DN'] if d in ds.used_descriptors()]
    used = ds.used_descriptors()
    if 'base_set' in st.session_state:
        st.session_state.base_set = [d for d in st.session_state.base_set if d in used]
    else:
        st.session_state.base_set = base_default
    sb.multiselect('Базовый набор дескрипторов', used, key='base_set', help=T['base_set'])

    sb.markdown('---')
    sb.markdown('## 🧪 Вещества в расчёте')
    q = sb.text_input('Поиск по названию', key='sb_search', placeholder='например, pyr')
    n_in, n_all, _ = ds.summary()
    sb.caption(f'Включено {n_in} из {n_all}')
    for _, r in ds.values.iterrows():
        name = str(r[D.NAME])
        if q and q.lower() not in name.lower():
            continue
        rid = int(r[D.ROW_ID])
        st.session_state[f'inc_{rid}'] = bool(r[D.INCLUDED])
        label = ('➕ ' if bool(r[D.ADDED]) else '') + name
        sb.checkbox(label, key=f'inc_{rid}', on_change=_toggle_one, args=(rid,))


def ransac_settings():
    u = units_key()
    try:
        grid = parse_grid(st.session_state.get(f'tau_grid_{u}', default_grid(u)))
    except ValueError as e:
        st.sidebar.error(f'Сетка τ: {e} Использую сетку по умолчанию.')
        grid = tuple(A.TAU_GRID_MMOL_G * D.UNITS[u]['tau_factor'])
    auto = st.session_state.get('tau_mode', 'Авто').startswith('Авто')
    return {
        'threshold': None if auto else float(st.session_state.get(f'tau_fixed_{u}', 1.0)),
        'tau_grid': grid,
        'min_fraction': st.session_state.get('min_fraction', 20) / 100,
        'min_floor': int(st.session_state.get('min_floor', 4)),
        'min_r2': float(st.session_state.get('min_r2', 0.5)),
        'min_cv_r2': float(st.session_state.get('min_q2', 0.5)),
        'n_iterations': int(st.session_state.get('max_lines', 5)),
        'max_trials': int(st.session_state.get('max_trials', 20000)),
        'seed': int(st.session_state.get('seed', 42)),
    }


def is_positive(descriptor):
    mode = st.session_state.get('positive_mode', 'По свойству')
    if mode.startswith('Требовать'):
        return True
    if mode.startswith('Не'):
        return False
    return ds.meta[descriptor].get('positive') == 'да'


def ci_level():
    return float(st.session_state.get('ci_level', 0.95)) if st.session_state.get('ci_on', True) else None


def ylabels():
    u = D.UNITS[units_key()]
    return u['col'], u['axis'], u['tau_factor']


# =============================================================================
# 📋 ДАННЫЕ
# =============================================================================
def editor_frame():
    v = ds.values
    orig = ds.original.set_index(D.ROW_ID)
    cols = [D.NAME, D.SORPTION] + ds.descriptors + [D.M_COL, D.RHO_COL, D.SMILES, D.CLASS_COL,
                                                    D.IR_COL, D.EXCEL_M, D.EXCEL_GG]
    out = pd.DataFrame({D.INCLUDED: v[D.INCLUDED].astype(bool)})
    status, was = [], []
    for _, r in v.iterrows():
        rid = r[D.ROW_ID]
        s = []
        if bool(r[D.ADDED]):
            s.append('➕ добавлено')
        if not bool(r[D.INCLUDED]):
            s.append('⏸ выключено')
        changes = []
        if not bool(r[D.ADDED]) and rid in orig.index:
            o = orig.loc[rid]
            for c in cols:
                if c not in orig.columns:
                    continue
                a, b = o[c], r[c]
                if (isinstance(a, float) and np.isnan(a)) and (b == '' or (isinstance(b, float) and np.isnan(b))):
                    continue
                if a != b:
                    changes.append(f'{c}: {D.fmt_number(a) if not isinstance(a, str) else a or "пусто"}')
        if changes:
            s.append('✏️ изменено')
        status.append(' '.join(s))
        was.append('; '.join(changes))
    out['Статус'] = status
    for c in cols:
        if c in ds.numeric_columns():
            out[c] = v[c].map(D.fmt_number)
        else:
            out[c] = v[c].fillna('').astype(str)
    out['Было в Excel'] = was
    return out


def _apply_editor():
    key = f'editor_{st.session_state.editor_v}'
    state = st.session_state.get(key, {})
    shown = st.session_state.get('editor_columns', [])
    errors, n = [], 0
    for pos, change in state.get('edited_rows', {}).items():
        rid = int(ds.values.iloc[int(pos)][D.ROW_ID])
        name = ds.name_of(rid)
        for col, val in change.items():
            try:
                if col == D.INCLUDED:
                    n += ds.set_included([rid], bool(val), 'таблица')
                elif col in shown:
                    n += int(ds.set_value(rid, col, val))
            except ValueError as e:
                errors.append(f'«{name}», столбец «{col}»: {e}')
    for e in errors:
        flash('error', f'Значение не принято — {e}')
    if n:
        flash('success', f'Изменений применено: {n}. Все вкладки пересчитаны.')
    st.session_state.editor_v += 1


def _apply_meta():
    key = f'meta_{st.session_state.meta_v}'
    state = st.session_state.get(key, {})
    fields = {'Использовать в анализе': 'use', 'Подпись на графике': 'label', 'Единицы': 'units',
              'Пояснение': 'explanation', 'Рост усиливает связывание': 'positive'}
    for pos, change in state.get('edited_rows', {}).items():
        d = ds.descriptors[int(pos)]
        for col, val in change.items():
            if col in fields:
                ds.set_meta(d, fields[col], bool(val) if fields[col] == 'use' else str(val))
    st.session_state.meta_v += 1


def _bulk(flag):
    n = ds.set_included(ds.values[D.ROW_ID].tolist(), flag, 'все')
    st.session_state.editor_v += 1
    flash('success', f'{"Включено" if flag else "Выключено"} веществ: {n}.')


def _preset(kind):
    n = ds.preset(kind)
    st.session_state.editor_v += 1
    flash('success', f'Набор применён, изменено веществ: {n}.')


def _class_toggle(cls, flag):
    ids = ds.values.loc[ds.values[D.CLASS_COL] == cls, D.ROW_ID].tolist()
    n = ds.set_included(ids, flag, f'класс «{cls}»')
    st.session_state.editor_v += 1
    flash('success', f'Класс «{cls}»: {"включено" if flag else "выключено"} веществ: {n}.')


def _undo():
    what = ds.undo()
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1
    flash('info' if what else 'warning', f'Отменено: {what}.' if what else 'Отменять нечего.')


def _reset():
    st.session_state.ds = new_dataset()
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1
    st.session_state.confirm_reset = False
    flash('success', 'Датасет сброшен к исходному Excel (набор «как в ноутбуке filtered»).')


def _calc_smiles():
    s = st.session_state.get('new_smiles', '')
    try:
        vals = D.descriptors_from_smiles(s)
    except Exception as e:
        flash('error', str(e))
        return
    for c, v in vals.items():
        st.session_state[f'new_{c}'] = D.fmt_number(v)
    flash('success', 'По SMILES рассчитаны: ' + ', '.join(f'{c} = {D.fmt_number(v)}' for c, v in vals.items())
          + '. TPSA 3D и Polar SASA 3D зависят от 3D-конформации — введите их вручную.')


def _add_substance():
    ss = st.session_state
    cls = ss.get('new_class_text', '').strip() if ss.get('new_class') == '➕ новый класс…' else ss.get('new_class', '')
    fields = {D.NAME: ss.get('new_name', ''), D.SORPTION: ss.get('new_sorption', ''),
              D.SMILES: ss.get('new_smiles', ''), D.CLASS_COL: cls or '',
              D.IR_COL: (ss.get('new_ir') or '')[:1].strip()}
    for c in ds.descriptors + [D.M_COL, D.RHO_COL]:
        fields[c] = ss.get(f'new_{c}', '')
    try:
        ds.add_substance(fields)
    except ValueError as e:
        flash('error', f'Вещество не добавлено: {e}')
        return
    for k in [k for k in ss.keys() if str(k).startswith('new_')]:
        del ss[k]
    ss.editor_v += 1
    flash('success', f'Вещество «{fields[D.NAME]}» добавлено и включено в расчёт.')


def _add_descriptor():
    ss = st.session_state
    try:
        ds.add_descriptor(ss.get('nd_name', ''), ss.get('nd_units', ''), ss.get('nd_expl', ''),
                          ss.get('nd_positive', 'не знаю'))
    except ValueError as e:
        flash('error', f'Дескриптор не добавлен: {e}')
        return
    flash('success', f'Дескриптор «{ss.get("nd_name")}» добавлен. Заполните значения в таблице или '
                     f'вставьте столбец ниже — он сразу появится в корреляциях, RANSAC и валидации.')
    for k in ('nd_name', 'nd_units', 'nd_expl'):
        ss.pop(k, None)
    ss.editor_v += 1
    ss.meta_v += 1


def _paste():
    ss = st.session_state
    try:
        n = ds.paste_column(ss['paste_col'], ss.get('paste_text', ''),
                            'order' if ss.get('paste_mode', '').startswith('Список') else 'pairs')
    except ValueError as e:
        flash('error', f'Вставка не выполнена: {e}')
        return
    ss.editor_v += 1
    ss.paste_text = ''
    flash('success', f'Вставлено значений: {n}.')


def _delete_added_substance():
    rid = st.session_state.get('del_sub')
    if rid is None:
        return
    try:
        name = ds.name_of(rid)
        ds.delete_substance(rid)
        flash('success', f'Вещество «{name}» удалено.')
    except (ValueError, KeyError) as e:
        flash('error', str(e))
    st.session_state.editor_v += 1


def _delete_added_descriptor():
    d = st.session_state.get('del_desc')
    if not d:
        return
    try:
        ds.delete_descriptor(d)
        flash('success', f'Дескриптор «{d}» удалён.')
    except ValueError as e:
        flash('error', str(e))
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1


def _upload():
    f = st.session_state.get('upload_file')
    if f is None:
        return
    try:
        st.session_state.ds = D.Dataset.from_xlsx(f)
        st.session_state.ds._log('Загружен сохранённый датасет', getattr(f, 'name', ''))
        flash('success', f'Датасет «{getattr(f, "name", "")}» загружен: восстановлены значения, '
                         'добавленные вещества и дескрипторы, включённые вещества и журнал.')
    except ValueError as e:
        flash('error', f'Файл не загружен: {e}')
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1


def page_data():
    st.markdown('### 📋 Датасет')
    st.warning('⚠️ ' + T['session'], icon=None)

    c = [st.container(horizontal=True)] * 4
    c[0].download_button('💾 Скачать мой датасет (xlsx)', data=lambda: ds.to_xlsx(),
                         file_name=f'graphene_dataset_{time.strftime("%Y-%m-%d_%H-%M")}.xlsx',
                         on_click='ignore', type='primary', key='dl_dataset')
    with c[1].popover('📂 Загрузить сохранённый датасет'):
        st.file_uploader('Файл, скачанный кнопкой «Скачать мой датасет»', type=['xlsx'],
                         key='upload_file')
        st.button('Загрузить', on_click=_upload, disabled=st.session_state.get('upload_file') is None)
    c[2].button('↩️ Отменить последнее изменение', on_click=_undo, disabled=not ds.undo_stack)
    with c[3].popover('🔄 Сбросить всё к исходному Excel'):
        st.write('Все правки, добавленные вещества и дескрипторы будут удалены. '
                 'Скачайте датасет, если он ещё нужен.')
        st.checkbox('Да, сбросить', key='confirm_reset')
        st.button('Сбросить', on_click=_reset, disabled=not st.session_state.get('confirm_reset'),
                  type='primary')

    # ---- включение и выключение
    st.markdown('#### Какие вещества участвуют в расчётах')
    c = [st.container(horizontal=True)] * 5
    c[0].button('✅ Включить все', on_click=_bulk, args=(True,))
    c[1].button('⛔ Выключить все', on_click=_bulk, args=(False,))
    c[2].button('Все вещества из Excel', on_click=_preset, args=('all',),
                help='Включить все исходные вещества из Excel (добавленные не меняются).')
    c[3].button('Как в ноутбуке filtered', on_click=_preset, args=('notebook',),
                help='Выключить n-бутиламин, морфолин, пиперидин, PBr3 и SOCl2 (точное совпадение '
                     'названий с учётом синонимов), остальные исходные — включить.')
    with st.expander('По химическим классам', expanded=False):
        explain('chem_class')
        for cls in ds.classes():
            members = ds.values[ds.values[D.CLASS_COL] == cls]
            n_on = int(members[D.INCLUDED].astype(bool).sum())
            cc = st.columns([3, 1, 1, 5])
            cc[0].markdown(f'**{cls}** — включено {n_on} из {len(members)}')
            cc[1].button('включить', key=f'clon_{cls}', on_click=_class_toggle, args=(cls, True))
            cc[2].button('выключить', key=f'cloff_{cls}', on_click=_class_toggle, args=(cls, False))
            cc[3].caption(', '.join(members[D.NAME].astype(str)))
        no_cls = ds.values.loc[ds.values[D.CLASS_COL].astype(str).str.strip() == '', D.NAME].tolist()
        if no_cls:
            st.caption('Без класса: ' + ', '.join(map(str, no_cls)))

    # ---- таблица
    st.markdown('#### Таблица')
    st.caption('Редактируйте любую ячейку двойным щелчком. Десятичный разделитель — запятая или точка. '
               'Пустая ячейка допустима. Изменённые строки помечены «✏️ изменено», исходное значение — '
               'в столбце «Было в Excel» справа. Чтобы скопировать, выделите ячейки и нажмите Ctrl/Cmd+C.')
    frame = editor_frame()
    editable = [D.NAME, D.SORPTION] + ds.descriptors + [D.M_COL, D.RHO_COL, D.SMILES, D.CLASS_COL,
                                                        D.IR_COL, D.EXCEL_M, D.EXCEL_GG]
    st.session_state.editor_columns = editable
    cfg = {
        D.INCLUDED: st.column_config.CheckboxColumn('Включено', pinned=True, width='small'),
        'Статус': st.column_config.TextColumn('Статус', disabled=True, width='small'),
        D.NAME: st.column_config.TextColumn('Name', pinned=True, width='medium'),
        D.SORPTION: st.column_config.TextColumn('Сорбция, ммоль/г', pinned=True, width='small'),
        D.IR_COL: st.column_config.SelectboxColumn(D.IR_COL, options=['', '1', '2', '3'],
                                                   help=T['ir_class']),
        D.M_COL: st.column_config.TextColumn(D.M_COL, help='Молярная масса; при загрузке рассчитана '
                                                           'по SMILES (RDKit). Используется для г/г и q·Vm.'),
        D.RHO_COL: st.column_config.TextColumn(D.RHO_COL, help='Плотность жидкости; нужна для q·Vm.'),
        D.EXCEL_M: st.column_config.TextColumn(D.EXCEL_M + ' (Excel)',
                                               help='Столбец из Excel. Испорчен (102.089 почти у всех), '
                                                    'в расчётах не используется.'),
        D.EXCEL_GG: st.column_config.TextColumn(D.EXCEL_GG + ' (Excel)',
                                                help='Столбец из Excel; в расчётах г/г пересчитывается '
                                                     'из ммоль/г и M.'),
        'Было в Excel': st.column_config.TextColumn('Было в Excel', disabled=True, width='large'),
    }
    for d in ds.descriptors:
        m = ds.meta[d]
        cfg[d] = st.column_config.TextColumn(('➕ ' if m.get('added') else '') + d,
                                             help=(m.get('explanation') or '') +
                                                  ('' if m.get('use', True) else ' (исключён из анализа)'))
    st.data_editor(frame, key=f'editor_{st.session_state.editor_v}', on_change=_apply_editor,
                   column_config=cfg, hide_index=True, num_rows='fixed', height=520,
                   disabled=['Статус', 'Было в Excel'], width='stretch')
    explain('missing', 'Что делать с пустыми ячейками?')

    # ---- добавление
    c1, c2 = st.columns(2)
    with c1.expander('➕ Добавить вещество', expanded=False):
        with st.form('add_sub', clear_on_submit=False):
            a, b = st.columns(2)
            a.text_input('Название *', key='new_name')
            b.text_input('Сорбция, ммоль/г *', key='new_sorption')
            a.selectbox('Химический класс', [''] + ds.classes() + ['➕ новый класс…'], key='new_class')
            b.text_input('Новый класс (если выбран «новый класс…»)', key='new_class_text')
            a.selectbox('Класс ИК-отклика', [''] + list(D.IR_CLASSES.values()), key='new_ir',
                        help=T['ir_class'])
            b.text_input('SMILES', key='new_smiles')
            st.markdown('**Дескрипторы** (можно оставить пустыми)')
            cols = st.columns(3)
            for i, cname in enumerate(ds.descriptors + [D.M_COL, D.RHO_COL]):
                cols[i % 3].text_input(cname, key=f'new_{cname}')
            s1, s2 = st.columns(2)
            s1.form_submit_button('🧮 Рассчитать дескрипторы по SMILES', on_click=_calc_smiles,
                                  disabled=not D.rdkit_available())
            s2.form_submit_button('Добавить вещество', on_click=_add_substance, type='primary')
        if not D.rdkit_available():
            st.caption('RDKit не установлен — дескрипторы по SMILES вводятся вручную.')
        else:
            st.caption('По SMILES рассчитываются молярная масса и TopoPSA 2D (RDKit, N/O) — так же '
                       'посчитан этот столбец в Excel. TPSA 3D (Mordred) и Polar SASA 3D зависят от '
                       '3D-конформации и вводятся вручную.')
        added = ds.values[ds.values[D.ADDED].astype(bool)]
        if len(added):
            st.selectbox('Удалить добавленное вещество', added[D.ROW_ID].tolist(), key='del_sub',
                         format_func=lambda r: ds.name_of(r))
            st.button('🗑 Удалить вещество', on_click=_delete_added_substance)
        st.caption('Исходные вещества из Excel не удаляются — их можно только выключить.')

    with c2.expander('➕ Добавить дескриптор', expanded=False):
        with st.form('add_desc'):
            st.text_input('Название (уникальное) *', key='nd_name')
            st.text_input('Единицы измерения', key='nd_units', placeholder='например, кДж/моль')
            st.text_area('Короткое пояснение: что означает дескриптор', key='nd_expl', height=80)
            st.radio('Рост дескриптора должен усиливать связывание?', D.POSITIVE_CHOICES, index=2,
                     key='nd_positive', horizontal=True, help=T['positive'])
            st.form_submit_button('Добавить дескриптор', on_click=_add_descriptor, type='primary')
        added_d = [d for d in ds.descriptors if ds.meta[d].get('added')]
        if added_d:
            st.selectbox('Удалить добавленный дескриптор', added_d, key='del_desc')
            st.button('🗑 Удалить дескриптор', on_click=_delete_added_descriptor)
        st.caption('Исходные дескрипторы не удаляются — их можно исключить из анализа в таблице '
                   '«Дескрипторы» ниже.')

    with st.expander('📥 Вставить столбец значений из Excel', expanded=False):
        st.selectbox('Куда вставить', ds.descriptors + [D.M_COL, D.RHO_COL], key='paste_col')
        st.radio('Формат', ['Список чисел по порядку строк таблицы',
                            'Таблица «название — значение» (два столбца из Excel)'], key='paste_mode')
        st.text_area('Вставьте сюда (Ctrl/Cmd+V)', key='paste_text', height=150,
                     placeholder='1,25\n0,8\n\nили\n\nPyridine\t0,57\nWater\t0,47')
        st.button('Вставить', on_click=_paste)

    st.markdown('#### Дескрипторы')
    meta_df = pd.DataFrame([{'Дескриптор': ('➕ ' if ds.meta[d].get('added') else '') + d,
                             'Использовать в анализе': bool(ds.meta[d].get('use', True)),
                             'Подпись на графике': ds.meta[d]['label'], 'Единицы': ds.meta[d]['units'],
                             'Рост усиливает связывание': ds.meta[d]['positive'],
                             'Пояснение': ds.meta[d]['explanation'],
                             'Заполнено': int(pd.to_numeric(ds.values[d], errors='coerce').notna().sum()),
                             'В ноутбуке': ds.meta[d].get('in_notebook', '')}
                            for d in ds.descriptors])
    st.data_editor(meta_df, key=f'meta_{st.session_state.meta_v}', on_change=_apply_meta, hide_index=True,
                   disabled=['Дескриптор', 'Заполнено', 'В ноутбуке'], width='stretch',
                   column_config={'Рост усиливает связывание': st.column_config.SelectboxColumn(
                       options=D.POSITIVE_CHOICES, help=T['positive']),
                       'Пояснение': st.column_config.TextColumn(width='large')})
    explain('positive', 'Что значит «рост усиливает связывание»?')

    with st.expander('🔍 Сравнить с исходным Excel', expanded=False):
        show_table(ds.diff(), 'differences_from_excel', 'diff', round_to=None)
        bad, n_bad_m = ds.excel_discrepancies()
        st.markdown(f'**Ошибки в самом Excel.** В столбце «{D.EXCEL_M}» у {n_bad_m} веществ значение не '
                    f'совпадает с молярной массой по SMILES (почти везде стоит 102.089), поэтому сайт '
                    f'использует M по SMILES. Вещества, у которых «{D.EXCEL_GG}» в Excel не равен '
                    f'ммоль/г × M / 1000:')
        show_table(bad, 'excel_gg_discrepancies', 'excel_bad')
    with st.expander(f'🕘 Журнал изменений ({len(ds.journal)})', expanded=False):
        show_table(pd.DataFrame(ds.journal[::-1]), 'change_log', 'journal', round_to=None)


# =============================================================================
# 🧩 КЛАСТЕРИЗАЦИЯ
# =============================================================================
@st.cache_data(show_spinner='Подбираю параметры кластеризации…', max_entries=50)
def cluster_candidates(Xs, seed):
    return {'kmeans': A.kmeans_candidates(Xs, seed=seed),
            'hc': A.hierarchical_candidates(Xs),
            'spectral': A.spectral_candidates(Xs, seed=seed),
            'dbscan': A.dbscan_candidates(Xs),
            'elbow': A.elbow_wcss(Xs, seed=seed)}


@st.cache_data(show_spinner=False, max_entries=200)
def c_kmeans(Xs, k, seed):
    return A.fit_kmeans(Xs, k, seed)


@st.cache_data(show_spinner=False, max_entries=200)
def c_hc(Xs, k, link):
    return A.fit_hierarchical(Xs, k, link)


@st.cache_data(show_spinner=False, max_entries=200)
def c_spectral(Xs, k, gamma, seed):
    return A.fit_spectral(Xs, k, gamma, seed)


@st.cache_data(show_spinner=False, max_entries=200)
def c_dbscan(Xs, eps, ms):
    return A.fit_dbscan(Xs, eps, ms)


def cand_table(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c.startswith('−'):
            df[c.lstrip('−')] = -df[c]
            df = df.drop(columns=c)
    return df


def page_clustering():
    st.markdown('### 🧩 Кластеризация')
    explain('clustering')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    feats = [d for d in ds.used_descriptors()]
    if not feats:
        st.error('Нет дескрипторов для кластеризации.')
        return
    xdesc = st.selectbox('Второй признак (в ноутбуке — ε)', feats,
                         index=feats.index('ε') if 'ε' in feats else 0, key='clust_feature')
    meta = ds.meta[xdesc]
    xlab = meta['label']
    data = frame[[A.NAME, ycol, xdesc, D.IR_COL]].copy()
    missing = data.loc[data[[ycol, xdesc]].isna().any(axis=1), A.NAME].tolist()
    data = data.dropna(subset=[ycol, xdesc]).reset_index(drop=True)
    used_note(len(data), missing, 'в кластеризацию')
    if len(data) < 5:
        st.error(f'Для кластеризации нужно хотя бы 5 веществ с сорбцией и «{xdesc}», сейчас {len(data)}.')
        return
    seed = int(st.session_state.get('seed', 42))
    X = data[[ycol, xdesc]].to_numpy(float)
    Xs = A.scale(X)
    cand = cluster_candidates(Xs, seed)
    n = len(data)
    kmax = max(2, min(8, n - 1))

    show_fig(P.raw_scatter(data[ycol], data[xdesc], xlab, meta.get('long_name') or xdesc, ycol),
             'sorption_vs_feature', 'raw', 900)
    with st.expander('Метод «локтя» (Elbow curve)'):
        explain('elbow')
        show_fig(P.elbow(*cand['elbow']), 'elbow', 'elbow', 900)

    results = {}
    tabs = st.tabs(['K-Means', 'Иерархическая', 'Спектральная', 'DBSCAN'])
    # ---- K-Means
    with tabs[0]:
        explain('kmeans')
        if not cand['kmeans']:
            st.error('Слишком мало веществ для k-средних.')
        else:
            auto_k = max(cand['kmeans'])[2]
            if st.checkbox('Автоматический выбор k (как в ноутбуке)', True, key='km_auto', help=T['auto_k']):
                k = auto_k
            else:
                k = st.slider('Число кластеров k', 2, kmax, min(auto_k, kmax), key='km_k')
            labels, centers = c_kmeans(Xs, k, seed)
            results['K-Means'] = (labels, f'k={k}', False)
            show_fig(P.cluster_scatter(X, labels, 'K-Means', yaxis, xlab,
                                       centers=StandardScaler().fit(X).inverse_transform(centers)),
                     'kmeans_clusters', 'km', 800)
            with st.expander('Перебор k: силуэт и Дэвис–Болдин'):
                show_table(cand_table(cand['kmeans'], ['Silhouette', '−Davies-Bouldin', 'k']),
                           'kmeans_candidates', 'kmc')
    # ---- Hierarchical
    with tabs[1]:
        explain('hierarchical')
        explain('linkage', 'Что такое linkage?')
        if not cand['hc']:
            st.error('Слишком мало веществ для иерархической кластеризации.')
        else:
            _, _, auto_k, auto_link = max(cand['hc'])
            if st.checkbox('Автоматический выбор k и linkage (как в ноутбуке)', True, key='hc_auto',
                           help=T['auto_k']):
                k, link = auto_k, auto_link
            else:
                c1, c2 = st.columns(2)
                k = c1.slider('Число кластеров k', 2, kmax, min(auto_k, kmax), key='hc_k')
                link = c2.selectbox('linkage', ['ward', 'complete', 'average'],
                                    index=['ward', 'complete', 'average'].index(auto_link), key='hc_link')
            labels = c_hc(Xs, k, link)
            results['Hierarchical'] = (labels, f'k={k}, linkage={link}', False)
            Z, leaves = A.dendrogram_linkage(data[A.NAME], X, link)
            show_fig(P.dendrogram(Z, leaves, A.dendrogram_threshold(Z, k, len(leaves))), 'dendrogram', 'dend', 1000)
            st.caption('Цвет подписей — группы из ноутбука (auto_group): красный — кислородсодержащие, '
                       'синий — азотсодержащие, оранжевый — серосодержащие, розовый — фосфорсодержащие, '
                       'чёрный — прочие.')
            show_fig(P.cluster_scatter(X, labels, 'Hierarchical', yaxis, xlab), 'hierarchical_clusters', 'hc', 800)
            with st.expander('Перебор linkage × k'):
                show_table(cand_table(cand['hc'], ['Silhouette', '−Davies-Bouldin', 'k', 'linkage']),
                           'hierarchical_candidates', 'hcc')
    # ---- Spectral
    with tabs[2]:
        explain('spectral')
        explain('gamma', 'Что такое γ?')
        Aff, sigma = A.adjacency_matrix(Xs)
        explain('adjacency', 'Как читать гистограмму матрицы смежности?')
        show_fig(P.adjacency_hist(Aff), 'adjacency_hist', 'adj', 1100)
        if n >= 3:
            ev, gaps, egk = A.laplacian_eigenvalues(Aff)
            explain('eigen', 'Как читать собственные значения?')
            show_fig(P.laplacian(ev), 'laplacian_eigenvalues', 'lap', 900)
            if len(gaps):
                st.caption(f'Eigengap подсказывает k = {egk}.')
                show_table(pd.DataFrame({'Number of clusters': np.arange(2, 2 + len(gaps)),
                                         'Difference between Eigenvalues': gaps}).T.reset_index(),
                           'eigengap', 'eg')
        if not cand['spectral']:
            st.error('Спектральная кластеризация не нашла допустимого разбиения.')
        else:
            _, _, auto_k, auto_g = max(cand['spectral'])
            if st.checkbox('Автоматический выбор k и γ (как в ноутбуке)', True, key='sp_auto', help=T['auto_k']):
                k, gamma = auto_k, auto_g
            else:
                c1, c2 = st.columns(2)
                k = c1.slider('Число кластеров k', 2, kmax, min(auto_k, kmax), key='sp_k')
                gamma = c2.select_slider('γ', list(A.SPECTRAL_GAMMAS), auto_g, key='sp_g')
            labels = c_spectral(Xs, k, gamma, seed)
            results['Spectral'] = (labels, f'k={k}, gamma={gamma}', False)
            show_fig(P.cluster_scatter(X, labels, 'Spectral Clustering', yaxis, xlab), 'spectral_clusters', 'sp', 800)
            with st.expander('Перебор γ × k'):
                show_table(cand_table(cand['spectral'], ['Silhouette', '−Davies-Bouldin', 'k', 'gamma']),
                           'spectral_candidates', 'spc')
        with st.expander('Подбор ширины ядра: StandardScaler и RobustScaler (ячейки 49–52)'):
            show_fig(P.adjacency_hist(A.adjacency_matrix(StandardScaler().fit_transform(X))[0],
                                      'Adjacency Matrix Value Distribution tuned by median heuristic (Standart Scaler)',
                                      (15, 5)), 'adjacency_standard', 'adjs', 1200)
            show_fig(P.adjacency_hist(A.adjacency_matrix(RobustScaler().fit_transform(X))[0],
                                      'Adjacency Matrix Value Distribution tuned by median heuristic (Robust Scaler)',
                                      (15, 5)), 'adjacency_robust', 'adjr', 1200)
    # ---- DBSCAN
    with tabs[3]:
        explain('dbscan')
        c1, c2 = st.columns(2)
        with c1:
            explain('eps', 'Что такое eps?')
        with c2:
            explain('min_samples', 'Что такое min_samples?')
        if cand['dbscan']:
            _, _, _, auto_eps, auto_ms = max(cand['dbscan'])
        else:
            auto_eps, auto_ms = 0.5, 4
            st.warning('Автоматический перебор не нашёл допустимого варианта (≥ 2 кластера, шум ≤ 45 %, '
                       'в каждом кластере ≥ 2 точки). Задайте параметры вручную.')
        auto = st.checkbox('Автоматический выбор eps и min_samples (как в ноутбуке)', bool(cand['dbscan']),
                           key='db_auto', help=T['auto_k'])
        if auto and cand['dbscan']:
            eps, ms = auto_eps, auto_ms
        else:
            c1, c2 = st.columns(2)
            eps = c1.select_slider('eps', list(A.DBSCAN_EPS_GRID), float(auto_eps), key='db_eps')
            ms = c2.slider('min_samples', 2, 12, int(auto_ms), key='db_ms')
        labels = c_dbscan(Xs, eps, ms)
        results['DBSCAN'] = (labels, f'eps={eps}, min_samples={ms}', True)
        show_fig(P.cluster_scatter(X, labels, 'DBSCAN', yaxis, xlab), 'dbscan_clusters', 'db', 800)
        with st.expander('Перебор eps × min_samples'):
            show_table(pd.DataFrame(cand['dbscan'], columns=['Score = silhouette − 0.2·noise', 'Silhouette',
                                                            '−Noise_fraction', 'eps', 'min_samples']),
                       'dbscan_candidates', 'dbc')

    if not results:
        return
    st.markdown('---')
    st.markdown('### Сводка по всем методам')
    metrics = pd.DataFrame([A.clustering_metrics(Xs, m, lab, par, noise)
                            for m, (lab, par, noise) in results.items()])
    show_table(metrics, 'clustering_metrics', 'metrics')
    c = st.columns(3)
    with c[0]:
        explain('silhouette', 'Силуэт')
    with c[1]:
        explain('ch', 'Калински–Харабаш')
    with c[2]:
        explain('db', 'Дэвис–Болдин')

    membership = pd.DataFrame({A.NAME: data[A.NAME], yaxis: data[ycol].round(3), xlab: data[xdesc].round(2)})
    for m, (lab, _, _) in results.items():
        membership[m] = np.where(np.asarray(lab) == -1, 'Noise', (np.asarray(lab) + 1).astype(str))
    st.markdown('**Какой зонд в какой кластер попал**')
    show_table(membership, 'cluster_membership', 'members', round_to=None)

    st.markdown('**Попарное сходство разбиений (ARI)**')
    explain('ari')
    show_table(A.ari_matrix({m: v[0] for m, v in results.items()}).reset_index(names='Метод'),
               'ari', 'ari', round_to=3)

    st.markdown('**Сорбция по кластерам**')
    explain('kruskal')
    ir = data[D.IR_COL].astype(str).str.strip()
    has_ir = (ir != '').sum() >= 3
    for m, (lab, _, _) in results.items():
        tab, H, p = A.kruskal_by_cluster(data[ycol], lab)
        st.markdown(f'*{m}*: критерий Краскела–Уоллиса H = {H:.3f}, p = {p:.4g}' if np.isfinite(p)
                    else f'*{m}*: критерий Краскела–Уоллиса не применим (нужно ≥ 2 кластера).')
        show_table(tab, f'sorption_by_cluster_{m}', f'kw_{m}')
    st.markdown('**Кластер × класс ИК-отклика**')
    if not has_ir:
        st.info('Классы ИК-отклика не заполнены (столбец «Класс ИК-отклика» на вкладке «Данные»). '
                'Таблица и критерии χ² / Фишера появятся, когда классы будут указаны хотя бы у трёх веществ.')
    else:
        explain('chi2')
        mask = ir != ''
        for m, (lab, _, _) in results.items():
            cl = np.where(np.asarray(lab) == -1, 'Noise', (np.asarray(lab) + 1).astype(str))
            res = A.contingency_test(cl[mask], ir[mask].to_numpy(), seed=seed)
            st.markdown(f'*{m}*: χ² = {res["chi2"]:.3f}, p = {res["chi2_p"]:.4g}; '
                        f'точный критерий Фишера ({res["fisher_kind"]}): p = {res["fisher_p"]:.4g}')
            show_table(res['table'].reset_index(names='Кластер'), f'cluster_x_ir_{m}', f'ir_{m}', round_to=None)

    # ---- RANSAC внутри кластеров
    st.markdown('---')
    st.markdown('### Последовательный RANSAC внутри кластеров')
    explain('cluster_ransac')
    rs = ransac_settings()
    tasks = {}
    tables = {}
    for m, (lab, _, _) in results.items():
        table = A.cluster_df(data[A.NAME], data[xdesc], data[ycol], lab, xlab, yaxis)
        table = table.sort_values('Cluster').reset_index(drop=True)
        tables[m] = table
        kw = dict(seed=rs['seed'], tau_grid=rs['tau_grid'], n_iterations=rs['n_iterations'],
                  min_fraction=rs['min_fraction'], min_floor=rs['min_floor'], min_r2=rs['min_r2'],
                  min_cv_r2=rs['min_cv_r2'], max_trials=rs['max_trials'], require_positive=True)
        tasks[('cluster', fhash(table, xlab, yaxis, kw))] = (A.analyze_cluster_groups, (table, xlab, yaxis), kw)
    done = compute_many(tasks)
    for (m, table), key in zip(tables.items(), tasks):
        groups = done[key]
        st.markdown(f'#### {m}')
        if not groups:
            st.caption('Нет кластеров для анализа.')
            continue
        show_fig(P.cluster_ransac(groups, m, xlab, yaxis, ci_level=ci_level(), tau_factor=tf),
                 f'{m}_cluster_ransac', f'cr_{m}', 1400)
        inl = [g['inliers'] for g in groups if not g['inliers'].empty]
        outl = [g['outliers'] for g in groups if not g['outliers'].empty]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('Inliers (на прямых)')
            show_table(pd.concat(inl, ignore_index=True) if inl else pd.DataFrame(), f'{m}_cluster_inliers',
                       f'cri_{m}')
        with c2:
            st.markdown('Outliers (вне прямых)')
            show_table(pd.concat(outl, ignore_index=True) if outl else pd.DataFrame(), f'{m}_cluster_outliers',
                       f'cro_{m}')


# =============================================================================
# 📈 КОРРЕЛЯЦИИ
# =============================================================================
def ransac_all(frame, descriptors):
    """RANSAC по каждому дескриптору; кэш по данным именно этого дескриптора."""
    ycol, _, _ = ylabels()
    rs = ransac_settings()
    tasks, keys = {}, {}
    for d in descriptors:
        sub = frame[[A.NAME, ycol, d]]
        kw = dict(min_fraction=rs['min_fraction'], min_floor=rs['min_floor'], n_iterations=rs['n_iterations'],
                  min_r2=rs['min_r2'], min_cv_r2=rs['min_cv_r2'], max_trials=rs['max_trials'],
                  threshold=rs['threshold'], require_positive=is_positive(d), seed=rs['seed'],
                  tau_grid=rs['tau_grid'])
        k = ('desc', fhash(sub, d, ycol, kw))
        tasks[k] = (A.analyze_descriptor, (sub, d, ycol), kw)
        keys[d] = k
    done = compute_many(tasks)
    return {d: done[keys[d]] for d in descriptors}


@st.cache_data(show_spinner='Строю плоскость…', max_entries=100)
def c_plane(sub, a, b, ycol, seed, tau_grid):
    return A.ransac_plane(sub, a, b, ycol, seed=seed, tau_grid=tau_grid)


def lines_summary(res_all):
    rows = []
    for d, r in res_all.items():
        for i, m in enumerate(r['models'], start=1):
            rows.append({'Descriptor': d, 'Line': i, 'n_inliers': len(m['inliers_idx']),
                         'N_total': len(r['data']), 'Slope': m['slope'], 'Intercept': m['intercept'],
                         'R2_in': m['r2'], 'Q2_LOO': m['q2'], 'tau': m['threshold'],
                         'Min_samples': m['min_samples'],
                         'Q_corr': m['r2'] * len(m['inliers_idx']) / len(r['data'])})
        if not r['models']:
            rows.append({'Descriptor': d, 'Line': '—', 'n_inliers': 0, 'N_total': len(r['data'])})
    return pd.DataFrame(rows)


def page_correlations():
    st.markdown('### 📈 Корреляции сорбции с дескрипторами')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    used = ds.used_descriptors()
    if not used:
        st.error('Нет дескрипторов, включённых в анализ.')
        return

    st.markdown('#### Пирсон и Спирмен (все вещества, без RANSAC)')
    explain('pearson')
    show_table(A.correlation_table(frame, used, ycol), 'pearson_spearman', 'corr')

    st.markdown('#### Последовательный RANSAC')
    explain('ransac')
    st.caption('Настройки τ, порогов R² и Q², числа прямых, max_trials и доверительной полосы — '
               'в боковой панели, «🎯 Настройки RANSAC». Знак наклона берётся из свойства '
               'дескриптора «рост усиливает связывание» (вкладка «Данные»).')
    res_all = ransac_all(frame, used)
    st.markdown('**Сводка по всем прямым**')
    show_table(lines_summary(res_all), 'ransac_lines_summary', 'lines')
    explain('q_corr', 'Что такое Q_corr и C_corr?')

    st.markdown('#### Два дескриптора рядом (как ячейка 57 ноутбука)')
    c1, c2 = st.columns(2)
    d1 = c1.selectbox('Левый график', used, index=used.index('β') if 'β' in used else 0, key='pair_a')
    d2 = c2.selectbox('Правый график', used, index=used.index('π*') if 'π*' in used else min(1, len(used) - 1),
                      key='pair_b')
    if d1 != d2:
        ok = all(res_all[d]['error'] is None for d in (d1, d2))
        if ok:
            kt = all(d in ('α', 'β', 'π*') for d in (d1, d2))
            title = 'Dependence of sorption on Kamlet-Taft parameters' if kt else \
                f'Dependence of sorption on {d1} and {d2}'
            show_fig(P.descriptor_pair({d1: res_all[d1], d2: res_all[d2]},
                                       [ds.meta[d1]['label'], ds.meta[d2]['label']], yaxis, title,
                                       ci_level=ci_level(), tau_factor=tf, ycol=ycol),
                     f'pair_{d1}_{d2}', 'pair', 1400)
        else:
            for d in (d1, d2):
                if res_all[d]['error']:
                    st.warning(res_all[d]['error'])

    st.markdown('#### Каждый дескриптор отдельно')
    chosen = st.multiselect('Показать дескрипторы', used, default=used, key='desc_show')
    for d in chosen:
        r = res_all[d]
        m = ds.meta[d]
        with st.container(border=True):
            st.markdown(f'##### {m["label"]}' + (f' — {m["units"]}' if m.get('units') and m['units'] != '—' else ''))
            if m.get('explanation'):
                with st.expander('ℹ️ Что это за дескриптор?'):
                    st.markdown(m['explanation'])
            used_note(len(r['data']), r['missing'])
            if r['error']:
                st.warning(r['error'])
                continue
            st.caption(f'Минимум точек на прямой: {r["min_inliers"]}. Положительный наклон: '
                       f'{"требуется" if is_positive(d) else "не требуется"}. Найдено прямых: {len(r["models"])}.')
            show_fig(P.descriptor_ransac(r['data'], d, m['label'], r['models'], yaxis, ci_level=ci_level(),
                                         tau_factor=tf, ycol=ycol), f'ransac_{d}', f'r_{d}', 1000)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown('Inliers (на прямых)')
                show_table(r.get('inliers', pd.DataFrame()), f'inliers_{d}', f'ri_{d}')
            with c2:
                st.markdown('Outliers (вне прямых)')
                show_table(r.get('outliers', pd.DataFrame()), f'outliers_{d}', f'ro_{d}')

    st.markdown('#### Класс ИК-отклика × на прямой / вне прямой')
    ir = frame[[A.NAME, D.IR_COL]].copy()
    ir[D.IR_COL] = ir[D.IR_COL].astype(str).str.strip()
    if (ir[D.IR_COL] != '').sum() < 3:
        st.info('Классы ИК-отклика не заполнены. Таблицы с точным критерием Фишера появятся, когда '
                'классы будут указаны (столбец «Класс ИК-отклика» на вкладке «Данные»).')
    else:
        explain('fisher_ir')
        ir_desc = st.multiselect('Дескрипторы', used, default=[d for d in ('β', 'DN') if d in used], key='ir_desc')
        irmap = dict(zip(ir[A.NAME], ir[D.IR_COL]))
        for d in ir_desc:
            r = res_all[d]
            on = {r['data'].iloc[i][A.NAME] for mm in r['models'] for i in mm['inliers_idx']}
            names = [n for n in r['data'][A.NAME] if irmap.get(n, '')]
            if len(names) < 3:
                st.caption(f'{d}: мало веществ с классом ИК.')
                continue
            res = A.contingency_test([irmap[n] for n in names],
                                     ['на прямой' if n in on else 'вне прямой' for n in names],
                                     seed=int(st.session_state.get('seed', 42)))
            st.markdown(f'*{d}*: точный критерий Фишера ({res["fisher_kind"]}) p = {res["fisher_p"]:.4g}; '
                        f'χ² p = {res["chi2_p"]:.4g}')
            show_table(res['table'].reset_index(names='Класс ИК'), f'ir_x_line_{d}', f'irl_{d}', round_to=None)

    st.markdown('#### RANSAC-плоскости по двум дескрипторам')
    explain('planes')
    base = st.session_state.get('base_set', [])
    c1, c2, c3 = st.columns([1, 1, 2])
    pa = c1.selectbox('Дескриптор X', used, index=used.index('β') if 'β' in used else 0, key='plane_a')
    pb = c2.selectbox('Дескриптор Y', used, index=used.index('π*') if 'π*' in used else min(1, len(used) - 1),
                      key='plane_b')
    mode = c3.radio('Какие вещества брать', ['Все дескрипторы базового набора заполнены (как в ноутбуке)',
                                             'Только выбранная пара заполнена'], key='plane_mode')
    if pa == pb:
        st.warning('Выберите два разных дескриптора.')
        return
    cols = [A.NAME, ycol, pa, pb]
    if mode.startswith('Все'):
        need = list(dict.fromkeys([ycol] + base + [pa, pb]))
        sub_all = frame[[A.NAME] + need]
        sub = sub_all.dropna()[cols]
        miss = sub_all.loc[sub_all.isna().any(axis=1), A.NAME].tolist()
        st.caption(f'Базовый набор: {", ".join(base) or "пуст"} (меняется в боковой панели).')
    else:
        sub = frame[cols].dropna()
        miss = frame.loc[frame[cols].isna().any(axis=1), A.NAME].tolist()
    used_note(len(sub), miss)
    rs = ransac_settings()
    res = c_plane(sub.reset_index(drop=True), pa, pb, ycol, rs['seed'], rs['tau_grid'])
    if res['error']:
        st.warning(res['error'])
        return
    show_fig(P.ransac_plane(res, pa, pb, ycol, tau_factor=tf), f'plane_{pa}_{pb}', 'plane', 750)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('Inliers (на плоскости)')
        show_table(res['inliers'], f'plane_inliers_{pa}_{pb}', 'pli')
    with c2:
        st.markdown('Outliers')
        show_table(res['outliers'], f'plane_outliers_{pa}_{pb}', 'plo')


# =============================================================================
# ✅ ВАЛИДАЦИЯ
# =============================================================================
@st.cache_data(show_spinner='Вложенная LOO-валидация PLS…', max_entries=50)
def c_pls(sub, predictors, ycol, seed):
    return A.pls_validation(sub, list(predictors), ycol, seed)


@st.cache_resource
def _unit_times():
    return {}


def time_estimate(frame, ycol, d, models, rs):
    """Замер одной перестановки каждого типа (один раз на конфигурацию)."""
    store = _unit_times()
    key = fhash(frame[[d, ycol]], d, rs)
    if key not in store:
        X, y = A.permutation_data(frame, d, ycol)
        mi = A.default_min_inliers(len(y), rs['min_fraction'], rs['min_floor'])
        t0 = time.time()
        for s in range(3):
            A.fixed_sequential_score(X, np.random.default_rng(s).permutation(y), models, mi, s)
        tc = (time.time() - t0) / 3
        st_ = dict(n_iterations=rs['n_iterations'], min_r2=rs['min_r2'], min_cv_r2=rs['min_cv_r2'],
                   require_positive=is_positive(d), tau_grid=rs['tau_grid'], threshold=rs['threshold'],
                   min_inliers=mi, max_trials_null=300)
        t0 = time.time()
        A._full_sequential_score(X, np.random.default_rng(0).permutation(y), st_, 0)
        tf_ = time.time() - t0
        store[key] = (tc, tf_)
    return store[key]


def page_validation():
    st.markdown('### ✅ Валидация')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    used = ds.used_descriptors()
    base = st.session_state.get('base_set', [])
    seed = int(st.session_state.get('seed', 42))

    # ---- коллинеарность
    st.markdown('#### Коллинеарность базового набора дескрипторов')
    explain('collinearity')
    st.caption(f'Базовый набор: {", ".join(base) or "пуст"} (меняется в боковой панели).')
    col = A.collinearity(frame, base, ycol)
    used_note(len(col['data']), col['missing'])
    if col['error']:
        st.warning(col['error'])
    else:
        show_fig(P.collinearity(col['corr'], col['evr']), 'collinearity', 'coll', 1300)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('VIF')
            show_table(col['vif'], 'vif', 'vif', round_to=3)
            explain('vif')
        with c2:
            st.markdown('Нагрузки PCA')
            show_table(col['loadings'].reset_index(names='Descriptor'), 'pca_loadings', 'pca', round_to=3)
            explain('pca')

    # ---- PLS
    st.markdown('#### Вложенная LOO-кросс-валидация PLS')
    explain('pls')
    pls = None
    if base:
        sub = frame[[A.NAME, ycol] + base]
        pls = c_pls(sub, tuple(base), ycol, seed)
        if pls.get('error'):
            st.warning(pls['error'])
            pls = None
        else:
            used_note(len(pls['y']), pls['missing'])
            show_fig(P.pls_observed_predicted(pls['y'], pls['predictions'], pls['q2'], pls['rmsecv'],
                                              D.UNITS[units_key()]['short']), 'nested_loo_pls', 'pls', 750)
            show_table(pd.DataFrame({'Complete_cases': [len(pls['y'])], 'Predictors': [len(base)],
                                     'Q2_LOO_nested': [pls['q2']], 'RMSECV': [pls['rmsecv']],
                                     'MAECV': [pls['maecv']]}), 'pls_metrics', 'plsm')
            show_table(pls['table'], 'pls_predictions', 'plst', round_to=3)

    # ---- перестановочные тесты
    st.markdown('---')
    st.markdown('#### Перестановочные тесты')
    explain('permutation')
    c1, c2 = st.columns(2)
    with c1:
        explain('perm_conditional', 'Условный тест')
    with c2:
        explain('perm_full', 'Полный тест')
    explain('s_corr', 'Что такое S_corr?')
    res_all = ransac_all(frame, used)
    with_lines = [d for d in used if res_all[d]['error'] is None and res_all[d]['models']]
    if not with_lines:
        st.info('Ни для одного дескриптора не найдено прямых RANSAC — тестировать нечего.')
    sel = st.multiselect('Дескрипторы (с найденными прямыми)', with_lines,
                         default=[d for d in with_lines], key='perm_desc')
    c1, c2, c3 = st.columns(3)
    n_cond = c1.slider('Перестановок, условный тест', 100, 2000, 2000, 100, key='n_cond')
    n_full = c2.slider('Перестановок, полный тест (0 — не считать)', 0, 2000, 200, 50, key='n_full')
    n_yr = c3.slider('Перестановок, y-рандомизация PLS (0 — не считать)', 0, 1000, 100, 50, key='n_yr')
    rs = ransac_settings()
    inputs_hash = fhash(frame, ycol, sel, n_cond, n_full, n_yr, rs, base,
                        [is_positive(d) for d in sel])
    if sel:
        tc, tfull = time_estimate(frame, ycol, sel[0], res_all[sel[0]]['models'], rs)
        workers = max(1, N_CPU)
        est = len(sel) * (n_cond * tc + n_full * tfull) / workers
        if n_yr and pls is not None:
            est += n_yr * 1.0 / workers
        st.caption(f'Оценка времени на этом сервере ({N_CPU} ядер): ~{est / 60:.1f} мин. '
                   f'Одна перестановка: условный тест ≈ {tc * 1000:.0f} мс, полный ≈ {tfull:.2f} с.')
    run = st.button('▶️ Рассчитать', type='primary', disabled=not sel)
    stored = st.session_state.get('validation')
    if run:
        out = {'hash': inputs_hash, 'cond': {}, 'full': {}, 'yr': None, 'sel': sel}
        bar = st.progress(0.0, text='Перестановочные тесты…')
        jobs = [(d, 'conditional', n_cond) for d in sel] + ([(d, 'full', n_full) for d in sel] if n_full else [])
        for j, (d, mode, nperm) in enumerate(jobs):
            def prog(f, j=j, d=d, mode=mode):
                bar.progress(min(1.0, (j + f) / (len(jobs) + (1 if n_yr else 0))),
                             text=f'{"Условный" if mode == "conditional" else "Полный"} тест: {d} ({int(f * 100)} %)')
            st_ = dict(n_iterations=rs['n_iterations'], min_r2=rs['min_r2'], min_cv_r2=rs['min_cv_r2'],
                       require_positive=is_positive(d), tau_grid=rs['tau_grid'], threshold=rs['threshold'])
            r = A.permutation_test(frame, d, res_all[d]['models'], ycol=ycol, n_permutations=nperm, seed=seed,
                                   mode=mode, settings=st_, progress=prog, min_fraction=rs['min_fraction'],
                                   min_floor=rs['min_floor'])
            out['cond' if mode == 'conditional' else 'full'][d] = r
        if n_yr and pls is not None:
            def prog(f):
                bar.progress(min(1.0, (len(jobs) + f) / (len(jobs) + 1)), text=f'y-рандомизация PLS ({int(f * 100)} %)')
            null = A.y_randomization(frame[[A.NAME, ycol] + base], base, ycol, n_yr, seed, progress=prog)
            out['yr'] = {'null': null, 'observed': pls['q2'],
                         'p': float((1 + np.sum(null >= pls['q2'])) / (n_yr + 1))}
        bar.empty()
        st.session_state.validation = out
        stored = out
    if not stored:
        st.info('Нажмите «Рассчитать». Результаты сохранятся до следующего изменения данных или настроек.')
        return
    if stored['hash'] != inputs_hash:
        st.error('⚠️ Результаты устарели: после расчёта изменились данные или настройки. Нажмите «Рассчитать» '
                 'ещё раз. Ниже показаны прежние результаты.')

    rows = []
    for d in stored['sel']:
        rc = stored['cond'].get(d)
        rf = stored['full'].get(d)
        rows.append({'Descriptor': d, 'Observed_S_corr': rc['observed'] if rc else np.nan,
                     'Null_mean (усл.)': rc['null'].mean() if rc else np.nan,
                     'Null_95_percentile (усл.)': np.quantile(rc['null'], 0.95) if rc else np.nan,
                     'p (условный)': rc['p'] if rc else np.nan, 'p (полный)': rf['p'] if rf else np.nan})
    summ = pd.DataFrame(rows)
    summ['p (условный), BH'] = A.benjamini_hochberg(summ['p (условный)'])
    summ['p (полный), BH'] = A.benjamini_hochberg(summ['p (полный)'])
    decisive = summ['p (полный), BH'].where(summ['p (полный), BH'].notna(), summ['p (условный), BH'])
    summ['Вывод (α = 0.05)'] = np.where(decisive < 0.05, 'значимо', 'не значимо')
    st.markdown('**Сводная таблица**')
    st.caption('Вывод делается по p после поправки Бенджамини–Хохберга: по полному тесту, если он посчитан, '
               'иначе по условному.')
    show_table(summ, 'permutation_summary', 'perm_sum')
    explain('bh', 'Что такое поправка Бенджамини–Хохберга?')
    if stored['cond']:
        st.markdown('**Условный тест: нулевые распределения**')
        show_fig(P.permutation_hists(stored['cond']), 'permutation_conditional', 'ph_c', 1300)
    if stored['full']:
        st.markdown('**Полный тест: нулевые распределения**')
        show_fig(P.permutation_hists(stored['full']), 'permutation_full', 'ph_f', 1300)
    if stored.get('yr'):
        st.markdown('**y-рандомизация PLS**')
        explain('y_random')
        yr = stored['yr']
        show_fig(P.y_randomization_hist(yr['null'], yr['observed'], yr['p']), 'y_randomization', 'yr', 750)
        st.caption(f'Q² настоящей модели = {yr["observed"]:.3f}; среднее Q² на перемешанных данных = '
                   f'{np.mean(yr["null"]):.3f}; p = {yr["p"]:.4f}.')


# =============================================================================
# Каркас страницы
# =============================================================================
st.title('Сорбция молекулярных зондов на оксиде графена')
with st.expander('ℹ️ Как пользоваться сайтом'):
    st.markdown(T['site'])
sidebar()
page = st.segmented_control('Раздел', PAGES, default=PAGES[0], key='page', label_visibility='collapsed')
page = page or PAGES[0]
headline()
show_flash()
{PAGES[0]: page_data, PAGES[1]: page_clustering, PAGES[2]: page_correlations,
 PAGES[3]: page_validation}[page]()
