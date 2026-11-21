"""Interactive site "Sorption of molecular probes on graphene oxide" (English version).

GENERATED from app.py by dev/make_en.py: edit app.py / dev/translations_en.py and regenerate.
Run:  streamlit run app_en.py
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
from streamlit import config as _st_config

import i18n
i18n.LANG = 'en'          # must be set before data / analysis are imported

# Dark-blue theme for this app only (the Russian app keeps .streamlit/config.toml).
# Streamlit re-sends the theme from the config at every script run, so setting it once
# and re-running applies it from the first page view.
_THEME = {
    'theme.base': 'light',
    'theme.primaryColor': '#1d4e89',
    'theme.textColor': '#14213d',
    'theme.backgroundColor': '#ffffff',
    'theme.secondaryBackgroundColor': '#edf2f9',
    'theme.linkColor': '#1d4e89',
    'theme.borderColor': '#cfd9e8',
    'theme.dataframeHeaderBackgroundColor': '#e3ebf6',
    'theme.sidebar.backgroundColor': '#0f2747',
    'theme.sidebar.textColor': '#e8eef7',
    'theme.sidebar.secondaryBackgroundColor': '#1a3a63',
    'theme.sidebar.primaryColor': '#8cb8ff',
    'theme.sidebar.borderColor': '#2c4d78',
    'theme.sidebar.linkColor': '#8cb8ff',
}
if _st_config.get_option('theme.primaryColor') != _THEME['theme.primaryColor']:
    for _k, _v in _THEME.items():
        _st_config.set_option(_k, _v, where_defined='app_en.py')
    st.rerun()
from sklearn.preprocessing import RobustScaler, StandardScaler

import analysis as A
import data as D
import plots as P
from explain_en import T

st.set_page_config(page_title='Graphene oxide probe sorption', page_icon='🧪', layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.headline {background: rgba(49,51,63,0.06); border-radius: 8px; padding: 0.55rem 0.9rem;
           margin-bottom: 0.8rem; font-size: 0.95rem;}
.small-note {color: #6b6f7b; font-size: 0.86rem;}
div[data-testid="stExpander"] summary p {font-size: 0.95rem;}

h1, h2, h3, h4 {color: #0f2747;}
h1 {font-weight: 800; letter-spacing: -0.5px;}
.headline {background: #edf2f9 !important; border-left: 4px solid #1d4e89; border-radius: 4px !important;}
[data-testid="stSidebar"] h2 {color: #ffffff;}
[data-testid="stExpander"] details {border-color: #cfd9e8;}
</style>
""", unsafe_allow_html=True)

def _effective_cpus():


    try:
        from joblib import cpu_count
        return max(1, int(cpu_count()))
    except Exception:
        return os.cpu_count() or 2


N_CPU = _effective_cpus()





_ON_CLOUD = str(os.path.abspath(__file__)).startswith('/mount/src')
N_WORKERS = max(1, min(N_CPU, int(os.environ.get('SITE_MAX_WORKERS', 1 if _ON_CLOUD else 4))))


def _free_workers():
    """Shut down joblib worker processes after a heavy run so they release memory."""
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
PAGES = ['📋 Data', '🧩 Clustering', '📈 Correlations', '✅ Validation']


# =============================================================================

# =============================================================================
@st.cache_data(show_spinner=False)
def _raw_excel():
    return D.read_excel_raw()


def new_dataset():
    ds = D.Dataset(_raw_excel())
    ds.preset('notebook')
    ds.journal.clear()
    ds.undo_stack.clear()
    ds._log('Initial set: as in the filtered notebook',
            'excluded n-butylamine, morpholine, piperidine, PBr3, SOCl2')
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

# =============================================================================
def explain(key, title='What does this mean?'):
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


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def line_styler(shown: pd.DataFrame, line_col='Line'):
    """Rows are tinted with the line colour; the line-number cell is filled with it."""
    colors = P.line_colors()

    def style(row):
        try:
            i = int(row[line_col]) - 1
        except (TypeError, ValueError):
            return [''] * len(row)
        if i < 0:
            return [''] * len(row)
        c = colors[i % len(colors)]
        return [f'background-color: {_rgba(c, 0.14)}'
                + (f'; color: white; background-color: {c}; font-weight: 700' if col == line_col else '')
                for col in row.index]
    fmt = {c: '{:.4g}' for c in shown.select_dtypes(include='float').columns}
    return shown.style.apply(style, axis=1).format(fmt, na_rep='')


def show_table(df: pd.DataFrame, name: str, key: str, height='auto', round_to=4, line_col=None, **kwargs):
    """Table with copy (select cells, Ctrl/Cmd+C) and CSV / Excel download.
    line_col: column with the RANSAC line number; rows are tinted with line colours."""
    if df is None or len(df) == 0:
        st.caption('— empty table —')
        return
    shown = df.copy()
    if round_to is not None:
        num = shown.select_dtypes(include='number').columns
        shown[num] = shown[num].round(round_to)
    if line_col and line_col in shown.columns:
        st.dataframe(line_styler(shown, line_col), hide_index=True, height=height, width='stretch', **kwargs)
    else:
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
    off_txt = ', '.join(off) if off else 'none'
    unit = D.UNITS[st.session_state.get('units', 'mmol/g')]['title']
    st.markdown(f'<div class="headline">In analysis: <b>{n_in}</b> of <b>{n_all}</b>; '
                f'excluded: {off_txt}. Sorption units: <b>{unit}</b>.</div>',
                unsafe_allow_html=True)


def used_note(n_used, missing, what='in analysis'):
    txt = f'Used {what}: **{n_used}** substances.'
    if missing:
        txt += f' Dropped for missing values ({len(missing)}): ' + ', '.join(missing) + '.'
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

# =============================================================================
@st.cache_resource
def _store():
    return OrderedDict()


def compute_many(tasks: dict) -> dict:
    """tasks: key → (function, args, kwargs). Returns key → result."""
    store = _store()
    missing = [k for k in tasks if k not in store]
    if missing:


        with st.spinner(f'Computing RANSAC ({len(missing)} task(s))…'):
            results = [tasks[k][0](*tasks[k][1], **tasks[k][2]) for k in missing]
        for k, r in zip(missing, results):
            store[k] = r
        while len(store) > 150:
            store.popitem(last=False)
    return {k: store[k] for k in tasks}


# =============================================================================

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
        raise ValueError('τ grid is empty.')
    return tuple(vals)


def _toggle_one(rid):
    ds.set_included([rid], bool(st.session_state[f'inc_{rid}']), 'sidebar')
    st.session_state.editor_v += 1


def sidebar():
    sb = st.sidebar
    sb.markdown('## ⚙️ Analysis settings')
    avail = ds.units_available()
    options = [u for u in D.UNITS if avail[u]]
    if units_key() not in options:
        st.session_state.units = 'mmol/g'
    sb.radio('Sorption units', options, key='units', format_func=lambda u: D.UNITS[u]['title'])
    if not avail['qVm']:
        sb.caption('q·Vm becomes available once density is filled in for all included substances '
                   '(column “Density, g/cm³” on the Data tab).')
    if not avail['g/g']:
        sb.caption('g/g unavailable: some included substances lack molar mass M.')
    with sb.expander('ℹ️ What does this mean?'):
        st.markdown(T['units'])
    sb.number_input('Random seed', min_value=0, max_value=2 ** 31 - 1, value=42, step=1,
                    key='seed')
    with sb.expander('ℹ️ Why a seed?'):
        st.markdown(T['seed'])

    with sb.expander('🎯 RANSAC settings', expanded=False):
        st.radio('Threshold τ', ['Auto (grid, as in notebook)', 'Fixed'], key='tau_mode')
        u = units_key()
        st.number_input(f'Fixed τ ({D.UNITS[u]["short"]})', min_value=0.0001,
                        value=1.0 * D.UNITS[u]['tau_factor'], step=0.05 * D.UNITS[u]['tau_factor'],
                        format='%.4f', key=f'tau_fixed_{u}',
                        disabled=st.session_state.get('tau_mode', '').startswith('Auto'))
        st.text_input(f'τ grid ({D.UNITS[u]["short"]})', value=default_grid(u), key=f'tau_grid_{u}',
                      help=T['tau_grid'])
        st.caption(T['tau'])
        st.slider('Min. share of points per line, %', 5, 60, 20, key='min_fraction', help=T['min_inliers'])
        st.number_input('…but at least, points', 2, 20, 4, key='min_floor')
        st.slider('Min. R²_in', 0.0, 0.99, 0.50, 0.01, key='min_r2', help=T['min_r2'])
        st.slider('Min. Q²_LOO', -1.0, 0.99, 0.50, 0.01, key='min_q2', help=T['min_q2'])
        st.slider('Max. lines', 1, 9, 5, key='max_lines', help=T['max_lines'])
        st.select_slider('max_trials', [500, 1000, 2000, 5000, 10000, 20000, 50000], 20000,
                         key='max_trials', help=T['max_trials'])
        st.radio('Positive slope', ['By descriptor property', 'Always require', 'Never require'],
                 key='positive_mode', help=T['positive'])
        st.checkbox('Confidence band on lines', True, key='ci_on', help=T['ci'])
        st.select_slider('Confidence level', [0.90, 0.95, 0.99], 0.95, key='ci_level',
                         format_func=lambda v: f'{int(v * 100)} %', disabled=not st.session_state.get('ci_on', True))

    base_default = [d for d in ['ε', 'α', 'β', 'π*', 'DN'] if d in ds.used_descriptors()]
    used = ds.used_descriptors()
    if 'base_set' in st.session_state:
        st.session_state.base_set = [d for d in st.session_state.base_set if d in used]
    else:
        st.session_state.base_set = base_default
    sb.multiselect('Base descriptor set', used, key='base_set', help=T['base_set'])

    sb.markdown('---')
    sb.markdown('## 🧪 Substances')
    q = sb.text_input('Search by name', key='sb_search', placeholder='e.g. pyr')
    n_in, n_all, _ = ds.summary()
    sb.caption(f'Included {n_in} of {n_all}')
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
        st.sidebar.error(f'τ grid: {e} Using the default grid.')
        grid = tuple(A.TAU_GRID_MMOL_G * D.UNITS[u]['tau_factor'])
    auto = st.session_state.get('tau_mode', 'Auto').startswith('Auto')
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
    mode = st.session_state.get('positive_mode', 'By descriptor')
    if mode.startswith('Always'):
        return True
    if mode.startswith('Never'):
        return False
    return ds.meta[descriptor].get('positive') == D.YES


def ci_level():
    return float(st.session_state.get('ci_level', 0.95)) if st.session_state.get('ci_on', True) else None


def ylabels():
    u = D.UNITS[units_key()]
    return u['col'], u['axis'], u['tau_factor']


# =============================================================================

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
            s.append('➕ added')
        if not bool(r[D.INCLUDED]):
            s.append('⏸ excluded')
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
                    changes.append(f'{c}: {D.fmt_number(a) if not isinstance(a, str) else a or 'empty'}')
        if changes:
            s.append('✏️ edited')
        status.append(' '.join(s))
        was.append('; '.join(changes))
    out['Status'] = status
    for c in cols:
        if c in ds.numeric_columns():
            out[c] = v[c].map(D.fmt_number)
        else:
            out[c] = v[c].fillna('').astype(str)
    out['Excel value'] = was
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
                    n += ds.set_included([rid], bool(val), 'table')
                elif col in shown:
                    n += int(ds.set_value(rid, col, val))
            except ValueError as e:
                errors.append(f'“{name}”, column “{col}”: {e}')
    for e in errors:
        flash('error', f'Value rejected — {e}')
    if n:
        flash('success', f'Changes applied: {n}. All tabs recomputed.')
    st.session_state.editor_v += 1


def _apply_meta():
    key = f'meta_{st.session_state.meta_v}'
    state = st.session_state.get(key, {})
    fields = {'Use in analysis': 'use', 'Plot label': 'label', 'Units': 'units',
              'Description': 'explanation', 'Increase strengthens binding': 'positive'}
    for pos, change in state.get('edited_rows', {}).items():
        d = ds.descriptors[int(pos)]
        for col, val in change.items():
            if col in fields:
                ds.set_meta(d, fields[col], bool(val) if fields[col] == 'use' else str(val))
    st.session_state.meta_v += 1


def _bulk(flag):
    n = ds.set_included(ds.values[D.ROW_ID].tolist(), flag, 'all')
    st.session_state.editor_v += 1
    flash('success', f'{'Included' if flag else 'Excluded'} substances: {n}.')


def _preset(kind):
    n = ds.preset(kind)
    st.session_state.editor_v += 1
    flash('success', f'Preset applied, substances changed: {n}.')


def _class_toggle(cls, flag):
    ids = ds.values.loc[ds.values[D.CLASS_COL] == cls, D.ROW_ID].tolist()
    n = ds.set_included(ids, flag, f'class “{cls}”')
    st.session_state.editor_v += 1
    flash('success', f'Class “{cls}”: {'included' if flag else 'excluded'} substances: {n}.')


def _undo():
    what = ds.undo()
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1
    flash('info' if what else 'warning', f'Undone: {what}.' if what else 'Nothing to undo.')


def _reset():
    st.session_state.ds = new_dataset()
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1
    st.session_state.confirm_reset = False
    flash('success', 'Dataset reset to the original Excel (filtered-notebook set).')


def _calc_smiles():
    s = st.session_state.get('new_smiles', '')
    try:
        vals = D.descriptors_from_smiles(s)
    except Exception as e:
        flash('error', str(e))
        return
    for c, v in vals.items():
        st.session_state[f'new_{c}'] = D.fmt_number(v)
    flash('success', 'Computed from SMILES: ' + ', '.join(f'{c} = {D.fmt_number(v)}' for c, v in vals.items())
          + '. TPSA 3D and Polar SASA 3D depend on the 3D conformer: enter them manually.')


def _add_substance():
    ss = st.session_state
    cls = ss.get('new_class_text', '').strip() if ss.get('new_class') == '➕ new class…' else ss.get('new_class', '')
    fields = {D.NAME: ss.get('new_name', ''), D.SORPTION: ss.get('new_sorption', ''),
              D.SMILES: ss.get('new_smiles', ''), D.CLASS_COL: cls or '',
              D.IR_COL: (ss.get('new_ir') or '')[:1].strip()}
    for c in ds.descriptors + [D.M_COL, D.RHO_COL]:
        fields[c] = ss.get(f'new_{c}', '')
    try:
        ds.add_substance(fields)
    except ValueError as e:
        flash('error', f'Substance not added: {e}')
        return
    for k in [k for k in ss.keys() if str(k).startswith('new_')]:
        del ss[k]
    ss.editor_v += 1
    flash('success', f'Substance “{fields[D.NAME]}” added and included.')


def _add_descriptor():
    ss = st.session_state
    try:
        ds.add_descriptor(ss.get('nd_name', ''), ss.get('nd_units', ''), ss.get('nd_expl', ''),
                          ss.get('nd_positive', 'unknown'))
    except ValueError as e:
        flash('error', f'Descriptor not added: {e}')
        return
    flash('success', f'Descriptor “{ss.get("nd_name")}” added. Fill in values in the table or '
                     f'paste a column below; it appears in correlations, RANSAC and validation at once.')
    for k in ('nd_name', 'nd_units', 'nd_expl'):
        ss.pop(k, None)
    ss.editor_v += 1
    ss.meta_v += 1


def _paste():
    ss = st.session_state
    try:
        n = ds.paste_column(ss['paste_col'], ss.get('paste_text', ''),
                            'order' if ss.get('paste_mode', '').startswith('List') else 'pairs')
    except ValueError as e:
        flash('error', f'Paste failed: {e}')
        return
    ss.editor_v += 1
    ss.paste_text = ''
    flash('success', f'Values pasted: {n}.')


def _delete_added_substance():
    rid = st.session_state.get('del_sub')
    if rid is None:
        return
    try:
        name = ds.name_of(rid)
        ds.delete_substance(rid)
        flash('success', f'Substance “{name}” deleted.')
    except (ValueError, KeyError) as e:
        flash('error', str(e))
    st.session_state.editor_v += 1


def _delete_added_descriptor():
    d = st.session_state.get('del_desc')
    if not d:
        return
    try:
        ds.delete_descriptor(d)
        flash('success', f'Descriptor “{d}” deleted.')
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
        st.session_state.ds._log('Saved dataset loaded', getattr(f, 'name', ''))
        flash('success', f'Dataset “{getattr(f, "name", "")}” loaded: values, '
                         'added substances and descriptors, inclusion and log restored.')
    except ValueError as e:
        flash('error', f'File not loaded: {e}')
    st.session_state.editor_v += 1
    st.session_state.meta_v += 1


def page_data():
    st.markdown('### 📋 Dataset')
    st.warning('⚠️ ' + T['session'], icon=None)

    c = [st.container(horizontal=True)] * 4
    c[0].download_button('💾 Download dataset (xlsx)', data=lambda: ds.to_xlsx(),
                         file_name=f'graphene_dataset_{time.strftime("%Y-%m-%d_%H-%M")}.xlsx',
                         on_click='ignore', type='primary', key='dl_dataset')
    with c[1].popover('📂 Load saved dataset'):
        st.file_uploader('File saved with “Download dataset”', type=['xlsx'],
                         key='upload_file')
        st.button('Load', on_click=_upload, disabled=st.session_state.get('upload_file') is None)
    c[2].button('↩️ Undo last change', on_click=_undo, disabled=not ds.undo_stack)
    with c[3].popover('🔄 Reset to original Excel'):
        st.write('All edits, added substances and descriptors will be lost. '
                 'Download the dataset first if you need it.')
        st.checkbox('Yes, reset', key='confirm_reset')
        st.button('Reset', on_click=_reset, disabled=not st.session_state.get('confirm_reset'),
                  type='primary')


    st.markdown('#### Substances in the analysis')
    c = [st.container(horizontal=True)] * 5
    c[0].button('✅ Include all', on_click=_bulk, args=(True,))
    c[1].button('⛔ Exclude all', on_click=_bulk, args=(False,))
    c[2].button('All Excel substances', on_click=_preset, args=('all',),
                help='Include all original Excel substances (added ones unchanged).')
    c[3].button('As in filtered notebook', on_click=_preset, args=('notebook',),
                help='Exclude n-butylamine, morpholine, piperidine, PBr3 and SOCl2 (exact name match incl. synonyms); '
                     'include all other originals.')
    with st.expander('By chemical class', expanded=False):
        explain('chem_class')
        for cls in ds.classes():
            members = ds.values[ds.values[D.CLASS_COL] == cls]
            n_on = int(members[D.INCLUDED].astype(bool).sum())
            cc = st.columns([3, 1, 1, 5])
            cc[0].markdown(f'**{cls}** — included {n_on} of {len(members)}')
            cc[1].button('include', key=f'clon_{cls}', on_click=_class_toggle, args=(cls, True))
            cc[2].button('exclude', key=f'cloff_{cls}', on_click=_class_toggle, args=(cls, False))
            cc[3].caption(', '.join(members[D.NAME].astype(str)))
        no_cls = ds.values.loc[ds.values[D.CLASS_COL].astype(str).str.strip() == '', D.NAME].tolist()
        if no_cls:
            st.caption('No class: ' + ', '.join(map(str, no_cls)))


    st.markdown('#### Table')
    st.caption('Double-click a cell to edit. Decimal separator: comma or dot. Empty cells are allowed. '
               'Edited rows are marked “✏️ edited”; the original value is in “Excel value”. '
               'Select cells and press Ctrl/Cmd+C to copy.')
    frame = editor_frame()
    editable = [D.NAME, D.SORPTION] + ds.descriptors + [D.M_COL, D.RHO_COL, D.SMILES, D.CLASS_COL,
                                                        D.IR_COL, D.EXCEL_M, D.EXCEL_GG]
    st.session_state.editor_columns = editable
    cfg = {
        D.INCLUDED: st.column_config.CheckboxColumn('Included', pinned=True, width='small'),
        'Status': st.column_config.TextColumn('Status', disabled=True, width='small'),
        D.NAME: st.column_config.TextColumn('Name', pinned=True, width='medium'),
        D.SORPTION: st.column_config.TextColumn('Sorption, mmol/g', pinned=True, width='small'),
        D.IR_COL: st.column_config.SelectboxColumn(D.IR_COL, options=['', '1', '2', '3'],
                                                   help=T['ir_class']),
        D.M_COL: st.column_config.TextColumn(D.M_COL, help='Molar mass, computed from SMILES (RDKit) on load. '
                                                           'Used for g/g and q·Vm.'),
        D.RHO_COL: st.column_config.TextColumn(D.RHO_COL, help='Liquid density; needed for q·Vm.'),
        D.EXCEL_M: st.column_config.TextColumn(D.EXCEL_M + ' (Excel)',
                                               help='From Excel. Corrupted (102.089 for most rows); '
                                                    'not used.'),
        D.EXCEL_GG: st.column_config.TextColumn(D.EXCEL_GG + ' (Excel)',
                                                help='From Excel; g/g is recomputed '
                                                     'from mmol/g and M.'),
        'Excel value': st.column_config.TextColumn('Excel value', disabled=True, width='large'),
    }
    for d in ds.descriptors:
        m = ds.meta[d]
        cfg[d] = st.column_config.TextColumn(('➕ ' if m.get('added') else '') + d,
                                             help=(m.get('explanation') or '') +
                                                  ('' if m.get('use', True) else ' (excluded from analysis)'))
    st.data_editor(frame, key=f'editor_{st.session_state.editor_v}', on_change=_apply_editor,
                   column_config=cfg, hide_index=True, num_rows='fixed', height=520,
                   disabled=['Status', 'Excel value'], width='stretch')
    explain('missing', 'Empty cells?')


    c1, c2 = st.columns(2)
    with c1.expander('➕ Add substance', expanded=False):
        with st.form('add_sub', clear_on_submit=False):
            a, b = st.columns(2)
            a.text_input('Name *', key='new_name')
            b.text_input('Sorption, mmol/g *', key='new_sorption')
            a.selectbox('Chemical class', [''] + ds.classes() + ['➕ new class…'], key='new_class')
            b.text_input('New class (if “new class…” selected)', key='new_class_text')
            a.selectbox('IR response class', [''] + list(D.IR_CLASSES.values()), key='new_ir',
                        help=T['ir_class'])
            b.text_input('SMILES', key='new_smiles')
            st.markdown('**Descriptors** (optional)')
            cols = st.columns(3)
            for i, cname in enumerate(ds.descriptors + [D.M_COL, D.RHO_COL]):
                cols[i % 3].text_input(cname, key=f'new_{cname}')
            s1, s2 = st.columns(2)
            s1.form_submit_button('🧮 Compute from SMILES', on_click=_calc_smiles,
                                  disabled=not D.rdkit_available())
            s2.form_submit_button('Add substance', on_click=_add_substance, type='primary')
        if not D.rdkit_available():
            st.caption('RDKit not installed: enter SMILES-based descriptors manually.')
        else:
            st.caption('From SMILES: molar mass and TopoPSA 2D (RDKit, N/O), computed as in the Excel file. '
                       'TPSA 3D (Mordred) and Polar SASA 3D depend on the '
                       '3D conformer and must be entered manually.')
        added = ds.values[ds.values[D.ADDED].astype(bool)]
        if len(added):
            st.selectbox('Delete added substance', added[D.ROW_ID].tolist(), key='del_sub',
                         format_func=lambda r: ds.name_of(r))
            st.button('🗑 Delete substance', on_click=_delete_added_substance)
        st.caption('Original Excel substances cannot be deleted, only excluded.')

    with c2.expander('➕ Add descriptor', expanded=False):
        with st.form('add_desc'):
            st.text_input('Name (unique) *', key='nd_name')
            st.text_input('Units', key='nd_units', placeholder='e.g. kJ/mol')
            st.text_area('Short description', key='nd_expl', height=80)
            st.radio('Does an increase strengthen binding?', D.POSITIVE_CHOICES, index=2,
                     key='nd_positive', horizontal=True, help=T['positive'])
            st.form_submit_button('Add descriptor', on_click=_add_descriptor, type='primary')
        added_d = [d for d in ds.descriptors if ds.meta[d].get('added')]
        if added_d:
            st.selectbox('Delete added descriptor', added_d, key='del_desc')
            st.button('🗑 Delete descriptor', on_click=_delete_added_descriptor)
        st.caption('Original descriptors cannot be deleted; exclude them in the '
                   '“Descriptors” table below.')

    with st.expander('📥 Paste a column from Excel', expanded=False):
        st.selectbox('Target column', ds.descriptors + [D.M_COL, D.RHO_COL], key='paste_col')
        st.radio('Format', ['List of numbers in table row order',
                            'Two columns: name — value'], key='paste_mode')
        st.text_area('Paste here (Ctrl/Cmd+V)', key='paste_text', height=150,
                     placeholder='1.25\n0.8\n\nor\n\nPyridine\t0.57\nWater\t0.47')
        st.button('Paste', on_click=_paste)

    st.markdown('#### Descriptors')
    meta_df = pd.DataFrame([{'Descriptor': ('➕ ' if ds.meta[d].get('added') else '') + d,
                             'Use in analysis': bool(ds.meta[d].get('use', True)),
                             'Plot label': ds.meta[d]['label'], 'Units': ds.meta[d]['units'],
                             'Increase strengthens binding': ds.meta[d]['positive'],
                             'Description': ds.meta[d]['explanation'],
                             'Filled': int(pd.to_numeric(ds.values[d], errors='coerce').notna().sum()),
                             'In notebook': ds.meta[d].get('in_notebook', '')}
                            for d in ds.descriptors])
    st.data_editor(meta_df, key=f'meta_{st.session_state.meta_v}', on_change=_apply_meta, hide_index=True,
                   disabled=['Descriptor', 'Filled', 'In notebook'], width='stretch',
                   column_config={'Increase strengthens binding': st.column_config.SelectboxColumn(
                       options=D.POSITIVE_CHOICES, help=T['positive']),
                       'Description': st.column_config.TextColumn(width='large')})
    explain('positive', 'What does “increase strengthens binding” mean?')

    with st.expander('🔍 Compare with original Excel', expanded=False):
        show_table(ds.diff(), 'differences_from_excel', 'diff', round_to=None)
        bad, n_bad_m = ds.excel_discrepancies()
        st.markdown(f'**Errors in the Excel file.** In “{D.EXCEL_M}” for {n_bad_m} substances the value does not '
                    f'match the molar mass from SMILES (mostly 102.089), so the site '
                    f'uses M from SMILES. Substances whose “{D.EXCEL_GG}” in Excel differs from '
                    f'mmol/g × M / 1000:')
        show_table(bad, 'excel_gg_discrepancies', 'excel_bad')
    with st.expander(f'🕘 Change log ({len(ds.journal)})', expanded=False):
        show_table(pd.DataFrame(ds.journal[::-1]), 'change_log', 'journal', round_to=None)


# =============================================================================

# =============================================================================
@st.cache_data(show_spinner='Selecting clustering parameters…', max_entries=50)
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


@st.cache_data(show_spinner=False, max_entries=200)
def c_kscan(Xs, method, seed, linkage='ward', gamma=1.0, kmax=8):
    return A.k_scan(Xs, method, k_max=kmax, seed=seed, linkage=linkage, gamma=gamma)


def k_control(prefix, auto_k, kmax, toggle_label='Optimal k (as in notebook)'):
    """"Optimal k" toggle + always-visible slider."""
    auto = st.toggle(toggle_label, True, key=f'{prefix}_auto', help=T['auto_k'])
    key = f'{prefix}_k'
    if auto or key not in st.session_state:
        st.session_state[key] = int(min(auto_k, kmax))
    st.session_state[key] = int(min(max(2, st.session_state[key]), kmax))
    k = st.slider('Number of clusters k', 2, kmax, key=key, disabled=auto,
                  help='Switch off the toggle above to set k manually.') if kmax > 2 else 2
    return int(k), auto


def _scan_styler(scan, k):
    rec = A.k_recommendation(scan)
    best_rows = {'Silhouette': scan['Silhouette'].idxmax(), 'Calinski-Harabasz': scan['Calinski-Harabasz'].idxmax(),
                 'Davies-Bouldin': scan['Davies-Bouldin'].idxmin()}

    def style(row):
        out = []
        for col in row.index:
            css = []
            if row['k'] == k:
                css.append('background-color: rgba(255, 196, 0, 0.28); font-weight: 600')
            if col in best_rows and row.name == best_rows[col]:
                css.append('background-color: rgba(46, 160, 67, 0.30); color: #136f2a; font-weight: 700')
            if col == 'k' and row['k'] == rec.get('notebook'):
                css.append('color: #136f2a; font-weight: 700')
            out.append('; '.join(css))
        return out
    shown = scan.copy()
    shown.insert(1, 'Chosen', np.where(shown['k'] == k, '◀ chosen', ''))
    shown.insert(2, 'Optimum', np.where(shown['k'] == rec.get('notebook'), '★ notebook rule', ''))
    return shown.style.apply(style, axis=1).format(
        {'Silhouette': '{:.3f}', 'Calinski-Harabasz': '{:.2f}', 'Davies-Bouldin': '{:.3f}'}, na_rep='—')


def k_advice(scan, k, prefix, extra='', elbow=False):
    """Hint: is the chosen k optimal by the metrics, plus a highlighted metrics table for all k."""
    if scan is None or scan.empty:
        return
    rec = A.k_recommendation(scan)
    if not rec:
        return
    row = scan.loc[scan['k'] == k]
    sil_k = float(row['Silhouette'].iloc[0]) if len(row) else float('nan')
    best_k = rec['notebook']
    sil_best = float(scan.loc[scan['k'] == best_k, 'Silhouette'].iloc[0])
    agree = [name for name, key in (('silhouette', 'Silhouette'), ('Calinski–Harabasz', 'Calinski-Harabasz'),
                                    ('Davies–Bouldin', 'Davies-Bouldin')) if rec[key] == k]
    if k == best_k:
        st.success(f'✅ k = {k} — optimal by the notebook rule (highest silhouette = {sil_k:.3f}).'
                   + (f' Also best by: {", ".join(agree)}.' if agree else '') + extra)
    else:
        lost = (1 - sil_k / sil_best) * 100 if sil_best else float('nan')
        st.warning(f'⚠️ k = {k} — not optimal by silhouette: {sil_k:.3f} vs {sil_best:.3f} at k = {best_k} '
                   f'(worse by {lost:.0f} %).'
                   + (f' But k = {k} is best by: {", ".join(agree)}.' if agree else '')
                   + ' Another k is fine if it fits the chemistry better.' + extra)
    st.caption(f'Best k per metric: silhouette → **{rec["Silhouette"]}**, Calinski–Harabasz → '
               f'**{rec["Calinski-Harabasz"]}**, Davies–Bouldin → **{rec["Davies-Bouldin"]}**'
               + ('; elbow: see the Elbow curve above.' if elbow else '.'))
    with st.expander('📊 Metrics for each k (green: best value, yellow: chosen k)'):
        st.dataframe(_scan_styler(scan, k), hide_index=True, width='stretch')
        c = st.columns(3)
        with c[0]:
            explain('silhouette', 'Silhouette')
        with c[1]:
            explain('ch', 'Calinski–Harabasz')
        with c[2]:
            explain('db', 'Davies–Bouldin')
        show_table(scan, f'metrics_by_k_{prefix}', f'scan_{prefix}', round_to=4)


def dbscan_advice(Xs, labels, eps, ms, best):
    m = A.clustering_metrics(Xs, 'DBSCAN', labels, '', noise=True)
    n_cl, noise = m['Clusters'], m['Noise_fraction']
    if best is None:
        st.info(f'Clusters: {n_cl}, noise: {noise:.0%}.')
        return
    score_best, sil_best, negnoise, beps, bms = best
    if np.isnan(m['Silhouette']):
        st.warning(f'⚠️ With eps = {eps}, min_samples = {ms} the number of clusters is {n_cl}: metrics undefined. '
                   f'Notebook optimum: eps = {beps}, min_samples = {bms}.')
        return
    score = m['Silhouette'] - 0.20 * noise
    if (eps, ms) == (beps, bms):
        st.success(f'✅ eps = {eps}, min_samples = {ms} — optimal by the notebook rule '
                   f'(silhouette − 0.2·noise share = {score:.3f}). Clusters: {n_cl}, noise: {noise:.0%}.')
    else:
        st.warning(f'⚠️ Score silhouette − 0.2·noise share = {score:.3f} vs {score_best:.3f} at the optimum '
                   f'eps = {beps}, min_samples = {bms}. Now clusters: {n_cl}, noise: {noise:.0%}.')


def cand_table(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c.startswith('−'):
            df[c.lstrip('−')] = -df[c]
            df = df.drop(columns=c)
    return df


def page_clustering():
    st.markdown('### 🧩 Clustering')
    explain('clustering')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    feats = [d for d in ds.used_descriptors()]
    if not feats:
        st.error('No descriptors for clustering.')
        return
    xdesc = st.selectbox('Second feature (notebook: ε)', feats,
                         index=feats.index('ε') if 'ε' in feats else 0, key='clust_feature')
    meta = ds.meta[xdesc]
    xlab = meta['label']
    data = frame[[A.NAME, ycol, xdesc, D.IR_COL]].copy()
    missing = data.loc[data[[ycol, xdesc]].isna().any(axis=1), A.NAME].tolist()
    data = data.dropna(subset=[ycol, xdesc]).reset_index(drop=True)
    used_note(len(data), missing, 'in clustering')
    if len(data) < 5:
        st.error(f'Clustering needs at least 5 substances with sorption and “{xdesc}”, now {len(data)}.')
        return
    seed = int(st.session_state.get('seed', 42))
    X = data[[ycol, xdesc]].to_numpy(float)
    Xs = A.scale(X)
    cand = cluster_candidates(Xs, seed)
    n = len(data)
    kmax = max(2, min(8, n - 1))

    show_fig(P.raw_scatter(data[ycol], data[xdesc], xlab, meta.get('long_name') or xdesc, ycol),
             'sorption_vs_feature', 'raw', 900)
    with st.expander('Elbow curve'):
        explain('elbow')
        show_fig(P.elbow(*cand['elbow']), 'elbow', 'elbow', 900)

    results = {}
    st.info('🎛️ Each method has its own cluster count: switch off “Optimal …” and move the slider. '
            'Plots, tables, metrics and the number of in-cluster RANSAC plots follow. '
            'The hint below the slider shows whether the choice is optimal.')
    tabs = st.tabs(['K-Means', 'Hierarchical', 'Spectral', 'DBSCAN'])
    # ---- K-Means
    with tabs[0]:
        explain('kmeans')
        if not cand['kmeans']:
            st.error('Too few substances for k-means.')
        else:
            auto_k = max(cand['kmeans'])[2]
            k, _ = k_control('km', auto_k, kmax)
            scan = c_kscan(Xs, 'kmeans', seed, kmax=kmax)
            k_advice(scan, k, 'km', elbow=True)
            labels, centers = c_kmeans(Xs, k, seed)
            results['K-Means'] = (labels, f'k={k}', False)
            show_fig(P.cluster_scatter(X, labels, 'K-Means', yaxis, xlab,
                                       centers=StandardScaler().fit(X).inverse_transform(centers)),
                     'kmeans_clusters', 'km', 800)
    # ---- Hierarchical
    with tabs[1]:
        explain('hierarchical')
        explain('linkage', 'What is linkage?')
        if not cand['hc']:
            st.error('Too few substances for hierarchical clustering.')
        else:
            _, _, auto_k, auto_link = max(cand['hc'])
            links = ['ward', 'complete', 'average']
            c1, c2 = st.columns([2, 1])
            with c1:
                k, auto = k_control('hc', auto_k, kmax, 'Optimal k and linkage (as in notebook)')
            if auto:
                st.session_state['hc_link'] = auto_link
            elif 'hc_link' not in st.session_state:
                st.session_state['hc_link'] = auto_link
            link = c2.selectbox('linkage', links, key='hc_link', disabled=auto)
            scan = c_kscan(Xs, 'hc', seed, linkage=link, kmax=kmax)
            note = '' if link == auto_link else (f' Best linkage by the notebook rule: **{auto_link}** '
                                                 f'(k = {auto_k}); chosen: {link}.')
            k_advice(scan, k, 'hc', extra=note)
            labels = c_hc(Xs, k, link)
            results['Hierarchical'] = (labels, f'k={k}, linkage={link}', False)
            Z, leaves = A.dendrogram_linkage(data[A.NAME], X, link)
            show_fig(P.dendrogram(Z, leaves, A.dendrogram_threshold(Z, k, len(leaves))), 'dendrogram', 'dend', 1000)
            st.caption('Label colours are the notebook groups (auto_group): red — O-containing, '
                       'blue — N-containing, orange — S-containing, pink — P-containing, '
                       'black — other. Branch colours show the cut into k clusters.')
            show_fig(P.cluster_scatter(X, labels, 'Hierarchical', yaxis, xlab), 'hierarchical_clusters', 'hc', 800)
            with st.expander('linkage × k search (as in notebook)'):
                show_table(cand_table(cand['hc'], ['Silhouette', '−Davies-Bouldin', 'k', 'linkage']),
                           'hierarchical_candidates', 'hcc')
    # ---- Spectral
    with tabs[2]:
        explain('spectral')
        explain('gamma', 'What is γ?')
        if not cand['spectral']:
            st.error('Spectral clustering found no valid partition.')
        else:
            _, _, auto_k, auto_g = max(cand['spectral'])
            c1, c2 = st.columns([2, 1])
            with c1:
                k, auto = k_control('sp', auto_k, kmax, 'Optimal k and γ (as in notebook)')
            if auto or 'sp_g' not in st.session_state:
                st.session_state['sp_g'] = auto_g
            gamma = c2.select_slider('γ', list(A.SPECTRAL_GAMMAS), key='sp_g', disabled=auto)
            scan = c_kscan(Xs, 'spectral', seed, gamma=gamma, kmax=kmax)
            note = '' if gamma == auto_g else (f' Best γ by the notebook rule: **{auto_g}** (k = {auto_k}); '
                                               f'chosen γ = {gamma}.')
            k_advice(scan, k, 'sp', extra=note)
            labels = c_spectral(Xs, k, gamma, seed)
            results['Spectral'] = (labels, f'k={k}, gamma={gamma}', False)
            show_fig(P.cluster_scatter(X, labels, 'Spectral Clustering', yaxis, xlab), 'spectral_clusters', 'sp', 800)
            with st.expander('γ × k search (as in notebook)'):
                show_table(cand_table(cand['spectral'], ['Silhouette', '−Davies-Bouldin', 'k', 'gamma']),
                           'spectral_candidates', 'spc')
        Aff, sigma = A.adjacency_matrix(Xs)
        with st.expander('Diagnostics: adjacency matrix and Laplacian eigenvalues'):
            explain('adjacency', 'Reading the adjacency histogram')
            show_fig(P.adjacency_hist(Aff), 'adjacency_hist', 'adj', 1100)
            if n >= 3:
                ev, gaps, egk = A.laplacian_eigenvalues(Aff)
                explain('eigen', 'Reading the eigenvalues')
                show_fig(P.laplacian(ev), 'laplacian_eigenvalues', 'lap', 900)
                if len(gaps):
                    st.caption(f'Eigengap suggests k = {egk}.')
                    show_table(pd.DataFrame({'Number of clusters': np.arange(2, 2 + len(gaps)),
                                             'Difference between Eigenvalues': gaps}).T.reset_index(),
                               'eigengap', 'eg')
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
            explain('eps', 'What is eps?')
        with c2:
            explain('min_samples', 'What is min_samples?')
        st.caption('DBSCAN has no k parameter: the number of clusters follows from eps and min_samples.')
        if cand['dbscan']:
            best_db = max(cand['dbscan'])
            _, _, _, auto_eps, auto_ms = best_db
        else:
            best_db = None
            auto_eps, auto_ms = 0.5, 4
            st.warning('The automatic search found no valid setting (≥ 2 clusters, noise ≤ 45 %, '
                       '≥ 2 points per cluster). Set parameters manually.')
        auto = st.toggle('Optimal eps and min_samples (as in notebook)', bool(cand['dbscan']), key='db_auto',
                         help=T['auto_k'], disabled=not cand['dbscan'])
        if (auto and cand['dbscan']) or 'db_eps' not in st.session_state:
            st.session_state['db_eps'] = float(auto_eps)
            st.session_state['db_ms'] = int(auto_ms)
        c1, c2 = st.columns(2)
        eps = c1.select_slider('eps', list(A.DBSCAN_EPS_GRID), key='db_eps', disabled=auto)
        ms = c2.slider('min_samples', 2, 12, key='db_ms', disabled=auto)
        labels = c_dbscan(Xs, eps, ms)
        results['DBSCAN'] = (labels, f'eps={eps}, min_samples={ms}', True)
        dbscan_advice(Xs, labels, eps, ms, best_db)
        show_fig(P.cluster_scatter(X, labels, 'DBSCAN', yaxis, xlab), 'dbscan_clusters', 'db', 800)
        with st.expander('eps × min_samples search (as in notebook)'):
            show_table(pd.DataFrame(cand['dbscan'], columns=['Score = silhouette − 0.2·noise', 'Silhouette',
                                                            '−Noise_fraction', 'eps', 'min_samples']),
                       'dbscan_candidates', 'dbc')

    if not results:
        return
    st.markdown('---')
    st.markdown('### Summary of all methods')
    metrics = pd.DataFrame([A.clustering_metrics(Xs, m, lab, par, noise)
                            for m, (lab, par, noise) in results.items()])
    show_table(metrics, 'clustering_metrics', 'metrics')
    c = st.columns(3)
    with c[0]:
        explain('silhouette', 'Silhouette')
    with c[1]:
        explain('ch', 'Calinski–Harabasz')
    with c[2]:
        explain('db', 'Davies–Bouldin')

    membership = pd.DataFrame({A.NAME: data[A.NAME], yaxis: data[ycol].round(3), xlab: data[xdesc].round(2)})
    for m, (lab, _, _) in results.items():
        membership[m] = np.where(np.asarray(lab) == -1, 'Noise', (np.asarray(lab) + 1).astype(str))
    st.markdown('**Cluster membership**')
    show_table(membership, 'cluster_membership', 'members', round_to=None)

    st.markdown('**Pairwise agreement (ARI)**')
    explain('ari')
    show_table(A.ari_matrix({m: v[0] for m, v in results.items()}).reset_index(names='Method'),
               'ari', 'ari', round_to=3)

    st.markdown('**Sorption by cluster**')
    explain('kruskal')
    ir = data[D.IR_COL].astype(str).str.strip()
    has_ir = (ir != '').sum() >= 3
    for m, (lab, _, _) in results.items():
        tab, H, p = A.kruskal_by_cluster(data[ycol], lab)
        st.markdown(f'*{m}*: Kruskal–Wallis H = {H:.3f}, p = {p:.4g}' if np.isfinite(p)
                    else f'*{m}*: Kruskal–Wallis not applicable (needs ≥ 2 clusters).')
        show_table(tab, f'sorption_by_cluster_{m}', f'kw_{m}')
    st.markdown('**Cluster × IR response class**')
    if not has_ir:
        st.info('IR response classes are empty (column “IR response class” on the Data tab). '
                'The table and χ² / Fisher tests appear once at least three substances have a class.')
    else:
        explain('chi2')
        mask = ir != ''
        for m, (lab, _, _) in results.items():
            cl = np.where(np.asarray(lab) == -1, 'Noise', (np.asarray(lab) + 1).astype(str))
            res = A.contingency_test(cl[mask], ir[mask].to_numpy(), seed=seed)
            st.markdown(f'*{m}*: χ² = {res["chi2"]:.3f}, p = {res["chi2_p"]:.4g}; '
                        f'Fisher exact test ({res["fisher_kind"]}): p = {res["fisher_p"]:.4g}')
            show_table(res['table'].reset_index(names='Cluster'), f'cluster_x_ir_{m}', f'ir_{m}', round_to=None)


    st.markdown('---')
    st.markdown('### Sequential RANSAC within clusters')
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
            st.caption('No clusters to analyse.')
            continue
        st.caption(f'Plots: {len(groups)} — one per cluster ({results[m][1]}). '
                   'Row colours in the inlier table match the line colours.')
        show_fig(P.cluster_ransac(groups, m, xlab, yaxis, ci_level=ci_level(), tau_factor=tf),
                 f'{m}_cluster_ransac', f'cr_{m}', 1400)
        inl = [g['inliers'] for g in groups if not g['inliers'].empty]
        outl = [g['outliers'] for g in groups if not g['outliers'].empty]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('Inliers (on lines)')
            show_table(pd.concat(inl, ignore_index=True) if inl else pd.DataFrame(), f'{m}_cluster_inliers',
                       f'cri_{m}', line_col='Line')
        with c2:
            st.markdown('Outliers (off lines)')
            show_table(pd.concat(outl, ignore_index=True) if outl else pd.DataFrame(), f'{m}_cluster_outliers',
                       f'cro_{m}')


# =============================================================================

# =============================================================================
def ransac_all(frame, descriptors):
    """RANSAC per descriptor; cached on that descriptor data only."""
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


@st.cache_data(show_spinner='Fitting plane…', max_entries=100)
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
            rows.append({'Descriptor': d, 'Line': None, 'n_inliers': 0, 'N_total': len(r['data'])})
    out = pd.DataFrame(rows)
    if 'Line' in out:
        out['Line'] = out['Line'].astype('Int64')
    return out


def page_correlations():
    st.markdown('### 📈 Sorption vs descriptors')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    used = ds.used_descriptors()
    if not used:
        st.error('No descriptors are used in the analysis.')
        return

    st.markdown('#### Pearson and Spearman (all substances, no RANSAC)')
    explain('pearson')
    show_table(A.correlation_table(frame, used, ycol), 'pearson_spearman', 'corr')

    st.markdown('#### Sequential RANSAC')
    explain('ransac')
    st.caption('τ, R² / Q² thresholds, number of lines, max_trials and confidence band: '
               'sidebar → “🎯 RANSAC settings”. Slope sign comes from the descriptor property '
               '“increase strengthens binding” (Data tab).')
    res_all = ransac_all(frame, used)
    st.markdown('**Summary of all lines**')
    show_table(lines_summary(res_all), 'ransac_lines_summary', 'lines', line_col='Line')
    explain('q_corr', 'What are Q_corr and C_corr?')

    st.markdown('#### Two descriptors side by side (notebook cell 57)')
    c1, c2 = st.columns(2)
    d1 = c1.selectbox('Left plot', used, index=used.index('β') if 'β' in used else 0, key='pair_a')
    d2 = c2.selectbox('Right plot', used, index=used.index('π*') if 'π*' in used else min(1, len(used) - 1),
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

    st.markdown('#### Each descriptor')
    chosen = st.multiselect('Show descriptors', used, default=used, key='desc_show')
    for d in chosen:
        r = res_all[d]
        m = ds.meta[d]
        with st.container(border=True):
            st.markdown(f'##### {m["label"]}' + (f' — {m["units"]}' if m.get('units') and m['units'] != '—' else ''))
            if m.get('explanation'):
                with st.expander('ℹ️ About this descriptor'):
                    st.markdown(m['explanation'])
            used_note(len(r['data']), r['missing'])
            if r['error']:
                st.warning(r['error'])
                continue
            st.caption(f'Min. points per line: {r["min_inliers"]}. Positive slope: '
                       f'{'required' if is_positive(d) else 'not required'}. Lines found: {len(r["models"])}.')
            show_fig(P.descriptor_ransac(r['data'], d, m['label'], r['models'], yaxis, ci_level=ci_level(),
                                         tau_factor=tf, ycol=ycol), f'ransac_{d}', f'r_{d}', 1000)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown('Inliers (on lines)')
                show_table(r.get('inliers', pd.DataFrame()), f'inliers_{d}', f'ri_{d}', line_col='Line')
            with c2:
                st.markdown('Outliers (off lines)')
                show_table(r.get('outliers', pd.DataFrame()), f'outliers_{d}', f'ro_{d}')

    st.markdown('#### IR response class × on / off line')
    ir = frame[[A.NAME, D.IR_COL]].copy()
    ir[D.IR_COL] = ir[D.IR_COL].astype(str).str.strip()
    if (ir[D.IR_COL] != '').sum() < 3:
        st.info('IR response classes are empty. Fisher tables appear once classes are filled in '
                '(column “IR response class” on the Data tab).')
    else:
        explain('fisher_ir')
        ir_desc = st.multiselect('Descriptors', used, default=[d for d in ('β', 'DN') if d in used], key='ir_desc')
        irmap = dict(zip(ir[A.NAME], ir[D.IR_COL]))
        for d in ir_desc:
            r = res_all[d]
            on = {r['data'].iloc[i][A.NAME] for mm in r['models'] for i in mm['inliers_idx']}
            names = [n for n in r['data'][A.NAME] if irmap.get(n, '')]
            if len(names) < 3:
                st.caption(f'{d}: too few substances with an IR class.')
                continue
            res = A.contingency_test([irmap[n] for n in names],
                                     ['on line' if n in on else 'off line' for n in names],
                                     seed=int(st.session_state.get('seed', 42)))
            st.markdown(f'*{d}*: Fisher exact test ({res["fisher_kind"]}) p = {res["fisher_p"]:.4g}; '
                        f'χ² p = {res["chi2_p"]:.4g}')
            show_table(res['table'].reset_index(names='IR class'), f'ir_x_line_{d}', f'irl_{d}', round_to=None)

    st.markdown('#### RANSAC planes (two descriptors)')
    explain('planes')
    base = st.session_state.get('base_set', [])
    c1, c2, c3 = st.columns([1, 1, 2])
    pa = c1.selectbox('Descriptor X', used, index=used.index('β') if 'β' in used else 0, key='plane_a')
    pb = c2.selectbox('Descriptor Y', used, index=used.index('π*') if 'π*' in used else min(1, len(used) - 1),
                      key='plane_b')
    mode = c3.radio('Substances used', ['All base-set descriptors filled (as in notebook)',
                                             'Only the chosen pair filled'], key='plane_mode')
    if pa == pb:
        st.warning('Choose two different descriptors.')
        return
    cols = [A.NAME, ycol, pa, pb]
    if mode.startswith('All'):
        need = list(dict.fromkeys([ycol] + base + [pa, pb]))
        sub_all = frame[[A.NAME] + need]
        sub = sub_all.dropna()[cols]
        miss = sub_all.loc[sub_all.isna().any(axis=1), A.NAME].tolist()
        st.caption(f'Base set: {", ".join(base) or 'empty'} (change in the sidebar).')
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
        st.markdown('Inliers (on plane)')
        show_table(res['inliers'], f'plane_inliers_{pa}_{pb}', 'pli')
    with c2:
        st.markdown('Outliers')
        show_table(res['outliers'], f'plane_outliers_{pa}_{pb}', 'plo')


# =============================================================================

# =============================================================================
@st.cache_data(show_spinner='Nested LOO PLS validation…', max_entries=50)
def c_pls(sub, predictors, ycol, seed):
    return A.pls_validation(sub, list(predictors), ycol, seed)


@st.cache_resource
def _unit_times():
    return {}


def time_estimate(frame, ycol, d, models, rs):
    """Time one permutation of each kind (once per configuration)."""
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
    st.markdown('### ✅ Validation')
    frame = ds.analysis_frame(units_key())
    ycol, yaxis, tf = ylabels()
    used = ds.used_descriptors()
    base = st.session_state.get('base_set', [])
    seed = int(st.session_state.get('seed', 42))


    st.markdown('#### Collinearity of the base descriptor set')
    explain('collinearity')
    st.caption(f'Base set: {", ".join(base) or 'empty'} (change in the sidebar).')
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
            st.markdown('PCA loadings')
            show_table(col['loadings'].reset_index(names='Descriptor'), 'pca_loadings', 'pca', round_to=3)
            explain('pca')

    # ---- PLS
    st.markdown('#### Nested LOO cross-validation (PLS)')
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


    st.markdown('---')
    st.markdown('#### Permutation tests')
    explain('permutation')
    c1, c2 = st.columns(2)
    with c1:
        explain('perm_conditional', 'Conditional test')
    with c2:
        explain('perm_full', 'Full test')
    explain('s_corr', 'What is S_corr?')
    res_all = ransac_all(frame, used)
    with_lines = [d for d in used if res_all[d]['error'] is None and res_all[d]['models']]
    if not with_lines:
        st.info('No RANSAC lines found for any descriptor: nothing to test.')
    sel = st.multiselect('Descriptors (with lines)', with_lines,
                         default=[d for d in with_lines], key='perm_desc')
    c1, c2, c3 = st.columns(3)
    n_cond = c1.slider('Permutations, conditional test', 100, 2000, 2000, 100, key='n_cond')
    n_full = c2.slider('Permutations, full test (0 = skip)', 0, 2000, 100, 50, key='n_full')
    n_yr = c3.slider('Permutations, PLS y-randomisation (0 = skip)', 0, 1000, 100, 50, key='n_yr')
    rs = ransac_settings()
    inputs_hash = fhash(frame, ycol, sel, n_cond, n_full, n_yr, rs, base,
                        [is_positive(d) for d in sel])
    if sel:
        tc, tfull = time_estimate(frame, ycol, sel[0], res_all[sel[0]]['models'], rs)
        workers = N_WORKERS

        est = len(sel) * (n_cond * tc + n_full * tfull / workers)
        if n_yr and pls is not None:
            est += n_yr * 0.3 / workers
        est_txt = f'~{est:.0f} s' if est < 90 else f'~{est / 60:.1f} min'
        st.caption(f'Estimated time on this server (parallel workers: {N_WORKERS}): {est_txt}. '
                   f'One permutation: conditional ≈ {tc * 1000:.1f} ms, full ≈ {tfull * 1000:.0f} ms.')
    run = st.button('▶️ Run', type='primary', disabled=not sel)
    stored = st.session_state.get('validation')
    if run:
        out = {'hash': inputs_hash, 'cond': {}, 'full': {}, 'yr': None, 'sel': sel}
        bar = st.progress(0.0, text='Permutation tests…')
        jobs = [(d, 'conditional', n_cond) for d in sel] + ([(d, 'full', n_full) for d in sel] if n_full else [])
        for j, (d, mode, nperm) in enumerate(jobs):
            def prog(f, j=j, d=d, mode=mode):
                bar.progress(min(1.0, (j + f) / (len(jobs) + (1 if n_yr else 0))),
                             text=f'{'Conditional' if mode == "conditional" else 'Full'} test: {d} ({int(f * 100)} %)')
            st_ = dict(n_iterations=rs['n_iterations'], min_r2=rs['min_r2'], min_cv_r2=rs['min_cv_r2'],
                       require_positive=is_positive(d), tau_grid=rs['tau_grid'], threshold=rs['threshold'])
            r = A.permutation_test(frame, d, res_all[d]['models'], ycol=ycol, n_permutations=nperm, seed=seed,
                                   mode=mode, settings=st_, progress=prog, min_fraction=rs['min_fraction'],
                                   min_floor=rs['min_floor'], n_jobs=1 if mode == 'conditional' else N_WORKERS)
            out['cond' if mode == 'conditional' else 'full'][d] = r
        if n_yr and pls is not None:
            def prog(f):
                bar.progress(min(1.0, (len(jobs) + f) / (len(jobs) + 1)), text=f'PLS y-randomisation ({int(f * 100)} %)')
            null = A.y_randomization(frame[[A.NAME, ycol] + base], base, ycol, n_yr, seed, progress=prog,
                                     n_jobs=N_WORKERS)
            out['yr'] = {'null': null, 'observed': pls['q2'],
                         'p': float((1 + np.sum(null >= pls['q2'])) / (n_yr + 1))}
        bar.empty()
        _free_workers()
        st.session_state.validation = out
        stored = out
    if not stored:
        st.info('Press “Run”. Results are kept until data or settings change.')
        return
    if stored['hash'] != inputs_hash:
        st.error('⚠️ Results are outdated: data or settings changed after the run. Press “Run” '
                 'again. Previous results are shown below.')

    rows = []
    for d in stored['sel']:
        rc = stored['cond'].get(d)
        rf = stored['full'].get(d)
        rows.append({'Descriptor': d, 'Observed_S_corr': rc['observed'] if rc else np.nan,
                     'Null_mean (cond.)': rc['null'].mean() if rc else np.nan,
                     'Null_95_percentile (cond.)': np.quantile(rc['null'], 0.95) if rc else np.nan,
                     'p (conditional)': rc['p'] if rc else np.nan, 'p (full)': rf['p'] if rf else np.nan})
    summ = pd.DataFrame(rows)
    summ['p (conditional), BH'] = A.benjamini_hochberg(summ['p (conditional)'])
    summ['p (full), BH'] = A.benjamini_hochberg(summ['p (full)'])
    decisive = summ['p (full), BH'].where(summ['p (full), BH'].notna(), summ['p (conditional), BH'])
    summ['Verdict (α = 0.05)'] = np.where(decisive < 0.05, 'significant', 'not significant')
    st.markdown('**Summary**')
    st.caption('Verdict uses BH-adjusted p: full test if computed, '
               'otherwise conditional.')
    show_table(summ, 'permutation_summary', 'perm_sum')
    explain('bh', 'What is the Benjamini–Hochberg adjustment?')
    if stored['cond']:
        st.markdown('**Conditional test: null distributions**')
        show_fig(P.permutation_hists(stored['cond']), 'permutation_conditional', 'ph_c', 1300)
    if stored['full']:
        st.markdown('**Full test: null distributions**')
        show_fig(P.permutation_hists(stored['full']), 'permutation_full', 'ph_f', 1300)
    if stored.get('yr'):
        st.markdown('**PLS y-randomisation**')
        explain('y_random')
        yr = stored['yr']
        show_fig(P.y_randomization_hist(yr['null'], yr['observed'], yr['p']), 'y_randomization', 'yr', 750)
        st.caption(f'Real-model Q² = {yr["observed"]:.3f}; mean Q² on shuffled data = '
                   f'{np.mean(yr["null"]):.3f}; p = {yr["p"]:.4f}.')


# =============================================================================

# =============================================================================
st.title('Sorption of molecular probes on graphene oxide')
with st.expander('ℹ️ How to use'):
    st.markdown(T['site'])
sidebar()
page = st.segmented_control('Section', PAGES, default=PAGES[0], key='page', label_visibility='collapsed')
page = page or PAGES[0]
headline()
show_flash()
{PAGES[0]: page_data, PAGES[1]: page_clustering, PAGES[2]: page_correlations,
 PAGES[3]: page_validation}[page]()
