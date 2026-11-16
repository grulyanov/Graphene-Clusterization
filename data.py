"""Текущий датасет сайта: значения, включённые вещества, метаданные дескрипторов,
журнал изменений, отмена, экспорт и импорт xlsx, сравнение с исходным Excel.

Excel читается один раз (load_excel кэшируется в app.py). Все расчёты берут данные
только из Dataset.analysis_frame().
"""
from __future__ import annotations

import copy
import io
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / 'Graphene.xlsx'
SHEET = 'Epsilon'

NAME = A.NAME
SORPTION = A.SORPTION_MMOL
SMILES = 'SMILES'
EXCEL_M = 'Molar Masses, g/mol'
EXCEL_GG = 'Sorption, g/g'
M_COL = 'M, г/моль'
RHO_COL = 'Плотность, г/см³'
CLASS_COL = 'Химический класс'
IR_COL = 'Класс ИК-отклика'
INCLUDED = 'Включено'
ADDED = 'Добавлено'
ROW_ID = 'ID'

# Служебные числовые столбцы (не дескрипторы)
SERVICE_NUMERIC = [EXCEL_M, EXCEL_GG, M_COL, RHO_COL]
SERVICE_TEXT = [SMILES, CLASS_COL, IR_COL]
IR_CLASSES = {'1': '1 — новые полосы (перенос протона / реакция)',
              '2': '2 — только сдвиги полос (водородная связь)',
              '3': '3 — нет отклика'}

UNITS = {
    'mmol/g': {'title': 'ммоль/г', 'col': 'Sorption, mmol/g', 'axis': 'Sorption (mmol/g)',
               'short': 'mmol/g', 'tau_factor': 1.0},
    'g/g': {'title': 'г/г', 'col': 'Sorption, g/g', 'axis': 'Sorption (g/g)',
            'short': 'g/g', 'tau_factor': 0.1},
    'qVm': {'title': 'объём q·Vm, см³/г', 'col': 'q·Vm, cm³/g', 'axis': 'q·Vm (cm³/g)',
            'short': 'cm³/g', 'tau_factor': 0.1},
}

POSITIVE_CHOICES = ['да', 'нет', 'не знаю']
FORMAT_VERSION = 'graphene-site-dataset-v1'


# ---------------------------------------------------------------------------
# RDKit
# ---------------------------------------------------------------------------
def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def descriptors_from_smiles(smiles: str) -> dict:
    """Молярная масса и TopoPSA 2D (RDKit, N/O) по SMILES. 3D-дескрипторы (TPSA 3D Mordred,
    Polar SASA 3D) зависят от конформации и вводятся вручную."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f'RDKit не смог разобрать SMILES «{smiles}».')
    return {M_COL: round(Descriptors.MolWt(mol), 3),
            'TopoPSA 2D (RDKit, N/O), Å²': round(rdMolDescriptors.CalcTPSA(mol), 2)}


# ---------------------------------------------------------------------------
# Разбор чисел: принимаем и запятую, и точку
# ---------------------------------------------------------------------------
def parse_number(value):
    """Возвращает float, np.nan для пустого значения или бросает ValueError."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(' ', '').replace(' ', '')
    if text in ('', '—', '-', 'nan', 'NaN', 'None'):
        return np.nan
    text = text.replace(',', '.')
    if not re.fullmatch(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?', text):
        raise ValueError(f'«{value}» — не число. Введите число, например 1,25 или 1.25.')
    return float(text)


def fmt_number(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    try:
        return f'{float(value):.10g}'
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Загрузка исходного Excel
# ---------------------------------------------------------------------------
def read_excel_raw(path=EXCEL_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = df.columns.astype(str).str.strip()
    return df


def default_descriptor_meta(column: str) -> dict:
    return {'label': column, 'long_name': column, 'units': '', 'explanation': '',
            'positive': 'не знаю', 'use': True, 'added': False, 'in_notebook': ''}


def load_config_meta() -> dict:
    path = ROOT / 'config' / 'descriptors.csv'
    meta = {}
    if path.exists():
        cfg = pd.read_csv(path, dtype=str).fillna('')
        for _, r in cfg.iterrows():
            m = default_descriptor_meta(r['column'])
            m.update({k: r[k] for k in ('label', 'long_name', 'units', 'explanation', 'positive',
                                         'in_notebook')})
            meta[r['column']] = m
    return meta


def load_classes() -> dict:
    path = ROOT / 'config' / 'probe_classes.csv'
    if not path.exists():
        return {}
    cfg = pd.read_csv(path, dtype=str).fillna('')
    return {A.probe_key(n): c for n, c in zip(cfg['Name'], cfg[CLASS_COL])}


class Dataset:
    """Состояние датасета. Всё хранится в простых pandas/dict, чтобы легко копировать для отмены."""

    def __init__(self, raw: pd.DataFrame):
        raw = raw.copy()
        cfg_meta = load_config_meta()
        classes = load_classes()
        # Дескрипторы: все числовые столбцы, кроме сорбции и служебных
        self.descriptors: list[str] = [c for c in raw.columns
                                       if c not in (NAME, SORPTION, SMILES, EXCEL_M, EXCEL_GG)
                                       and pd.api.types.is_numeric_dtype(raw[c])]
        self.meta = {c: cfg_meta.get(c, default_descriptor_meta(c)) for c in self.descriptors}
        values = raw.copy()
        values[M_COL] = np.nan
        if rdkit_available():
            for i, s in values[SMILES].items():
                try:
                    values.at[i, M_COL] = descriptors_from_smiles(s)[M_COL]
                except Exception:
                    pass
        values[RHO_COL] = np.nan
        values[CLASS_COL] = [classes.get(A.probe_key(n), '') for n in values[NAME]]
        values[IR_COL] = ''
        values[INCLUDED] = True
        values[ADDED] = False
        values[ROW_ID] = np.arange(len(values))
        self.values = values.reset_index(drop=True)
        self.original = self.values.copy()
        self.original_meta = copy.deepcopy(self.meta)
        self.original_descriptors = list(self.descriptors)
        self.journal: list[dict] = []
        self.undo_stack: list[dict] = []
        self.next_id = len(values)
        self.version = 0

    # ------------------------------------------------------------------ state
    def _snapshot(self):
        return {'values': self.values.copy(), 'meta': copy.deepcopy(self.meta),
                'descriptors': list(self.descriptors), 'journal': list(self.journal),
                'next_id': self.next_id}

    def _push(self):
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _log(self, action, substance='', column='', old='', new=''):
        self.journal.append({'Время': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             'Действие': action, 'Вещество': substance, 'Столбец': column,
                             'Было': fmt_number(old) if not isinstance(old, str) else old,
                             'Стало': fmt_number(new) if not isinstance(new, str) else new})
        self.version += 1

    def undo(self) -> str | None:
        if not self.undo_stack:
            return None
        last = self.journal[-1]['Действие'] if self.journal else 'изменение'
        snap = self.undo_stack.pop()
        self.values, self.meta = snap['values'], snap['meta']
        self.descriptors, self.journal = snap['descriptors'], snap['journal']
        self.next_id = snap['next_id']
        self.version += 1
        return last

    def reset(self):
        self.__init__(read_excel_raw())

    # ---------------------------------------------------------------- helpers
    def numeric_columns(self):
        return [SORPTION] + self.descriptors + [M_COL, RHO_COL, EXCEL_M, EXCEL_GG]

    def row_index(self, row_id):
        idx = self.values.index[self.values[ROW_ID] == row_id]
        if len(idx) == 0:
            raise KeyError(row_id)
        return idx[0]

    def name_of(self, row_id):
        return str(self.values.at[self.row_index(row_id), NAME])

    def used_descriptors(self):
        return [d for d in self.descriptors if self.meta[d].get('use', True)]

    def classes(self):
        return sorted({c for c in self.values[CLASS_COL] if str(c).strip()})

    # ------------------------------------------------------------------ edits
    def set_value(self, row_id, column, raw_value):
        """Правка ячейки. Числа — с запятой или точкой; нечисловой ввод отклоняется."""
        i = self.row_index(row_id)
        name = self.name_of(row_id)
        old = self.values.at[i, column]
        if column == NAME:
            new = str(raw_value).strip()
            if not new:
                raise ValueError('Название не может быть пустым.')
            if new in set(self.values.loc[self.values[ROW_ID] != row_id, NAME].astype(str)):
                raise ValueError(f'Вещество «{new}» уже есть в таблице.')
        elif column in self.numeric_columns():
            new = parse_number(raw_value)
            if column == SORPTION and np.isnan(new):
                raise ValueError('Сорбция обязательна и не может быть пустой.')
        elif column == IR_COL:
            new = str(raw_value or '').strip()
            if new and new not in IR_CLASSES:
                raise ValueError('Класс ИК-отклика — 1, 2, 3 или пусто.')
        else:
            new = '' if raw_value is None else str(raw_value).strip()
        same = (old == new) or (isinstance(old, float) and isinstance(new, float)
                                and np.isnan(old) and np.isnan(new))
        if same:
            return False
        self._push()
        self.values.at[i, column] = new
        self._log('Изменено значение', name, column, old, new)
        return True

    def set_included(self, row_ids, flag: bool, reason=''):
        row_ids = [r for r in row_ids if bool(self.values.at[self.row_index(r), INCLUDED]) != flag]
        if not row_ids:
            return 0
        self._push()
        for r in row_ids:
            self.values.at[self.row_index(r), INCLUDED] = flag
        names = ', '.join(self.name_of(r) for r in row_ids)
        self._log(('Включено' if flag else 'Выключено') + (f' ({reason})' if reason else ''), names,
                  INCLUDED, str(not flag), str(flag))
        return len(row_ids)

    def preset(self, kind: str):
        """'all' — все вещества из Excel; 'notebook' — как в ноутбуке filtered."""
        orig = ~self.values[ADDED].astype(bool)
        target = orig.copy()
        if kind == 'notebook':
            target = orig & ~A.notebook_excluded_mask(self.values[NAME])
        changes = self.values.index[(self.values[INCLUDED].astype(bool) != target) & orig]
        if len(changes) == 0:
            return 0
        self._push()
        self.values.loc[orig, INCLUDED] = target[orig]
        self._log('Набор: ' + ('все вещества из Excel' if kind == 'all' else 'как в ноутбуке filtered'),
                  f'изменено веществ: {len(changes)}')
        return len(changes)

    def add_substance(self, fields: dict):
        name = str(fields.get(NAME, '')).strip()
        if not name:
            raise ValueError('Укажите название вещества.')
        if name in set(self.values[NAME].astype(str)):
            raise ValueError(f'Вещество «{name}» уже есть в таблице.')
        sorption = parse_number(fields.get(SORPTION))
        if np.isnan(sorption):
            raise ValueError('Укажите сорбцию в ммоль/г.')
        row = {c: np.nan for c in self.values.columns}
        row.update({NAME: name, SORPTION: sorption, SMILES: str(fields.get(SMILES, '') or '').strip(),
                    CLASS_COL: str(fields.get(CLASS_COL, '') or '').strip(),
                    IR_COL: str(fields.get(IR_COL, '') or '').strip(),
                    INCLUDED: True, ADDED: True, ROW_ID: self.next_id})
        for c in self.descriptors + [M_COL, RHO_COL]:
            if c in fields:
                row[c] = parse_number(fields[c])
        self._push()
        self.values = pd.concat([self.values, pd.DataFrame([row])], ignore_index=True)
        self.values[INCLUDED] = self.values[INCLUDED].astype(bool)
        self.values[ADDED] = self.values[ADDED].astype(bool)
        self.next_id += 1
        self._log('Добавлено вещество', name)

    def delete_substance(self, row_id):
        i = self.row_index(row_id)
        if not bool(self.values.at[i, ADDED]):
            raise ValueError('Исходные вещества из Excel не удаляются — их можно только выключить.')
        name = self.name_of(row_id)
        self._push()
        self.values = self.values.drop(index=i).reset_index(drop=True)
        self._log('Удалено вещество', name)

    def add_descriptor(self, name, units='', explanation='', positive='не знаю'):
        name = str(name).strip()
        if not name:
            raise ValueError('Укажите название дескриптора.')
        if name in self.values.columns or name in (INCLUDED, ADDED, ROW_ID):
            raise ValueError(f'Столбец «{name}» уже существует.')
        label = f'{name} ({units})' if units else name
        self._push()
        pos = list(self.values.columns).index(M_COL)
        self.values.insert(pos, name, np.nan)
        self.descriptors.append(name)
        self.meta[name] = {'label': label, 'long_name': name, 'units': units,
                           'explanation': explanation, 'positive': positive, 'use': True,
                           'added': True, 'in_notebook': ''}
        self._log('Добавлен дескриптор', '', name)

    def delete_descriptor(self, name):
        if not self.meta[name].get('added'):
            raise ValueError('Исходные дескрипторы не удаляются — их можно только исключить из анализа.')
        self._push()
        self.values = self.values.drop(columns=[name])
        self.descriptors.remove(name)
        del self.meta[name]
        self._log('Удалён дескриптор', '', name)

    def set_meta(self, name, key, value):
        old = self.meta[name].get(key)
        if old == value:
            return False
        self._push()
        self.meta[name][key] = value
        titles = {'use': 'Использовать в анализе', 'positive': 'Рост усиливает связывание',
                  'label': 'Подпись на графике', 'units': 'Единицы', 'explanation': 'Пояснение'}
        self._log(f'Свойство дескриптора: {titles.get(key, key)}', '', name, str(old), str(value))
        return True

    def paste_column(self, column, text, mode='order'):
        """Вставка столбца из буфера: 'order' — список чисел по порядку строк таблицы;
        'pairs' — строки «название <Tab/;> значение». Возвращает число заполненных ячеек."""
        lines = [ln for ln in str(text).replace('\r', '').split('\n')]
        while lines and not lines[-1].strip():
            lines.pop()
        updates = {}
        if mode == 'order':
            if len(lines) > len(self.values):
                raise ValueError(f'Вставлено {len(lines)} значений, а веществ в таблице {len(self.values)}.')
            for i, ln in enumerate(lines):
                try:
                    updates[self.values.at[i, ROW_ID]] = parse_number(ln)
                except ValueError as e:
                    raise ValueError(f'Строка {i + 1}: {e}')
        else:
            keys = {A.probe_key(n): rid for n, rid in zip(self.values[NAME], self.values[ROW_ID])}
            unknown = []
            for i, ln in enumerate(lines, start=1):
                if not ln.strip():
                    continue
                parts = re.split(r'\t|;', ln)
                if len(parts) < 2:
                    raise ValueError(f'Строка {i}: нужен формат «название<Tab>значение» или «название;значение».')
                name, val = parts[0].strip(), parts[-1]
                rid = keys.get(A.probe_key(name))
                if rid is None:
                    unknown.append(name)
                    continue
                try:
                    updates[rid] = parse_number(val)
                except ValueError as e:
                    raise ValueError(f'Строка {i}: {e}')
            if unknown:
                raise ValueError('Не найдены вещества: ' + ', '.join(unknown))
        if not updates:
            return 0
        self._push()
        for rid, v in updates.items():
            self.values.at[self.row_index(rid), column] = v
        self._log('Вставлен столбец', f'значений: {len(updates)}', column)
        return len(updates)

    def set_class_bulk(self, mapping: dict):
        self._push()
        for rid, cls in mapping.items():
            self.values.at[self.row_index(rid), CLASS_COL] = cls
        self._log('Изменены химические классы', f'веществ: {len(mapping)}', CLASS_COL)

    # ------------------------------------------------------------ for analysis
    def units_available(self):
        inc = self.values[self.values[INCLUDED].astype(bool)]
        out = {'mmol/g': True}
        out['g/g'] = bool(inc[M_COL].notna().all())
        out['qVm'] = bool(inc[M_COL].notna().all() and inc[RHO_COL].notna().all() and len(inc))
        return out

    def analysis_frame(self, units='mmol/g') -> pd.DataFrame:
        """Включённые вещества; первый столбец после имени — сорбция в выбранных единицах."""
        inc = self.values[self.values[INCLUDED].astype(bool)].reset_index(drop=True)
        ycol = UNITS[units]['col']
        q = pd.to_numeric(inc[SORPTION], errors='coerce')
        if units == 'mmol/g':
            y = q
        elif units == 'g/g':
            y = q * pd.to_numeric(inc[M_COL], errors='coerce') / 1000
        else:
            y = q * pd.to_numeric(inc[M_COL], errors='coerce') / pd.to_numeric(inc[RHO_COL], errors='coerce') / 1000
        frame = pd.DataFrame({NAME: inc[NAME].astype(str), ycol: y.astype(float)})
        for d in self.descriptors:
            frame[d] = pd.to_numeric(inc[d], errors='coerce').astype(float)
        frame[CLASS_COL] = inc[CLASS_COL].astype(str)
        frame[IR_COL] = inc[IR_COL].astype(str)
        return frame

    def summary(self):
        inc = self.values[INCLUDED].astype(bool)
        off = self.values.loc[~inc, NAME].astype(str).tolist()
        return int(inc.sum()), len(self.values), off

    # ------------------------------------------------------------- comparison
    def diff(self) -> pd.DataFrame:
        rows = []
        orig = self.original.set_index(ROW_ID)
        for _, r in self.values.iterrows():
            rid = r[ROW_ID]
            if bool(r[ADDED]) or rid not in orig.index:
                rows.append({'Вещество': r[NAME], 'Столбец': '—', 'В Excel': '—',
                             'Сейчас': 'добавлено на сайте'})
                continue
            o = orig.loc[rid]
            for c in self.values.columns:
                if c in (ROW_ID, ADDED) or c not in orig.columns:
                    continue
                a, b = o[c], r[c]
                both_nan = (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
                if both_nan or a == b:
                    continue
                if (isinstance(a, float) and np.isnan(a)) and b == '':
                    continue
                rows.append({'Вещество': r[NAME], 'Столбец': c,
                             'В Excel': fmt_number(a) if not isinstance(a, (str, bool, np.bool_)) else str(a),
                             'Сейчас': fmt_number(b) if not isinstance(b, (str, bool, np.bool_)) else str(b)})
        for d in self.descriptors:
            if self.meta[d].get('added'):
                rows.append({'Вещество': '—', 'Столбец': d, 'В Excel': '—',
                             'Сейчас': 'дескриптор добавлен на сайте'})
        for d in self.original_descriptors:
            if d not in self.descriptors:
                continue
            if not self.meta[d].get('use', True):
                rows.append({'Вещество': '—', 'Столбец': d, 'В Excel': 'в анализе',
                             'Сейчас': 'исключён из анализа'})
        return pd.DataFrame(rows, columns=['Вещество', 'Столбец', 'В Excel', 'Сейчас'])

    def excel_discrepancies(self) -> pd.DataFrame:
        """Расхождения молярной массы и г/г между Excel и расчётом по SMILES."""
        v = self.values
        q = pd.to_numeric(v[SORPTION], errors='coerce')
        gg_calc = q * v[M_COL] / 1000
        out = pd.DataFrame({'Вещество': v[NAME], 'M в Excel, г/моль': v[EXCEL_M],
                            'M по SMILES, г/моль': v[M_COL], 'г/г в Excel': v[EXCEL_GG],
                            'г/г из ммоль/г и M': gg_calc})
        bad_m = (out['M в Excel, г/моль'] - out['M по SMILES, г/моль']).abs() > 0.05
        bad_g = (out['г/г в Excel'] - out['г/г из ммоль/г и M']).abs() > 0.002
        out['Расхождение'] = np.where(bad_g, 'г/г в Excel не равен ммоль/г × M / 1000', '')
        return out[bad_g].reset_index(drop=True), int(bad_m.sum())

    # ------------------------------------------------------------ export/import
    def to_xlsx(self) -> bytes:
        buf = io.BytesIO()
        data = self.values.copy()
        data[INCLUDED] = data[INCLUDED].map(lambda x: 'да' if bool(x) else 'нет')
        data[ADDED] = data[ADDED].map(lambda x: 'да' if bool(x) else 'нет')
        meta = pd.DataFrame([{'Дескриптор': d, 'Подпись на графике': m['label'],
                              'Полное название': m['long_name'], 'Единицы': m['units'],
                              'Пояснение': m['explanation'],
                              'Рост усиливает связывание': m['positive'],
                              'Использовать в анализе': 'да' if m.get('use', True) else 'нет',
                              'Добавлен на сайте': 'да' if m.get('added') else 'нет',
                              'В ноутбуке': m.get('in_notebook', '')}
                             for d, m in ((d, self.meta[d]) for d in self.descriptors)])
        info = pd.DataFrame({'Ключ': ['format', 'saved_at'],
                             'Значение': [FORMAT_VERSION, datetime.now().isoformat(timespec='seconds')]})
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            data.to_excel(w, sheet_name='Датасет', index=False)
            meta.to_excel(w, sheet_name='Дескрипторы', index=False)
            pd.DataFrame(self.journal, columns=['Время', 'Действие', 'Вещество', 'Столбец', 'Было',
                                                'Стало']).to_excel(w, sheet_name='Журнал', index=False)
            info.to_excel(w, sheet_name='Инфо', index=False)
            # Точные значения (Excel хранит ~15 знаков): нужны, чтобы загрузка восстанавливала
            # датасет бит в бит. Если ячейку правили в Excel, берётся видимое значение.
            exact = self.values[[ROW_ID] + [c for c in self.numeric_columns() if c in self.values]].copy()
            for c in exact.columns[1:]:
                exact[c] = exact[c].map(lambda v: '' if pd.isna(v) else repr(float(v)))
            exact.to_excel(w, sheet_name='_exact', index=False)
            w.book['_exact'].sheet_state = 'hidden'
        return buf.getvalue()

    @classmethod
    def from_xlsx(cls, file) -> 'Dataset':
        try:
            book = pd.read_excel(file, sheet_name=None)
        except Exception as e:
            raise ValueError(f'Не удалось прочитать файл как Excel (.xlsx): {e}')
        need = {'Датасет', 'Дескрипторы', 'Журнал', 'Инфо'}
        if not need.issubset(book):
            raise ValueError('Это не файл, скачанный с сайта: нужны листы ' + ', '.join(sorted(need))
                             + f'. В файле: {", ".join(book)}.')
        info = book['Инфо']
        if FORMAT_VERSION not in set(info.get('Значение', pd.Series(dtype=object)).astype(str)):
            raise ValueError('Неизвестная версия формата файла.')
        data, meta = book['Датасет'], book['Дескрипторы']
        for c in (NAME, SORPTION, INCLUDED, ADDED):
            if c not in data.columns:
                raise ValueError(f'На листе «Датасет» нет столбца «{c}».')
        ds = cls(read_excel_raw())
        ds.descriptors = [str(d) for d in meta['Дескриптор']]
        ds.meta = {}
        for _, r in meta.iterrows():
            d = str(r['Дескриптор'])
            if d not in data.columns:
                raise ValueError(f'Дескриптор «{d}» описан, но столбца с ним нет.')
            ds.meta[d] = {'label': str(r['Подпись на графике']), 'long_name': str(r['Полное название']),
                          'units': '' if pd.isna(r['Единицы']) else str(r['Единицы']),
                          'explanation': '' if pd.isna(r['Пояснение']) else str(r['Пояснение']),
                          'positive': str(r['Рост усиливает связывание']),
                          'use': str(r['Использовать в анализе']) == 'да',
                          'added': str(r['Добавлен на сайте']) == 'да',
                          'in_notebook': '' if pd.isna(r.get('В ноутбуке')) else str(r.get('В ноутбуке'))}
        values = data.copy()
        for c in ds.numeric_columns():
            if c in values.columns:
                try:
                    values[c] = values[c].map(parse_number).astype(float)
                except ValueError as e:
                    raise ValueError(f'Столбец «{c}»: {e}')
        for c in (SMILES, CLASS_COL, IR_COL):
            if c not in values.columns:
                values[c] = ''
            values[c] = values[c].fillna('').astype(str).replace('nan', '')
        values[IR_COL] = values[IR_COL].str.replace(r'\.0$', '', regex=True)
        values[INCLUDED] = values[INCLUDED].astype(str).str.lower().isin(['да', 'true', '1'])
        values[ADDED] = values[ADDED].astype(str).str.lower().isin(['да', 'true', '1'])
        if values[NAME].isna().any() or values[NAME].astype(str).duplicated().any():
            raise ValueError('Названия веществ должны быть заполнены и не повторяться.')
        # исходные строки получают те же ID, что в Excel (для сравнения), добавленные — новые
        if ROW_ID in values.columns and values[ROW_ID].notna().all():
            values[ROW_ID] = values[ROW_ID].astype(int)
            if '_exact' in book:
                exact = book['_exact'].set_index(ROW_ID)
                for c in exact.columns:
                    if c not in values.columns:
                        continue
                    for i, rid in values[ROW_ID].items():
                        if rid not in exact.index:
                            continue
                        e, v = exact.at[rid, c], values.at[i, c]
                        if isinstance(e, str) and e and pd.notna(v):
                            e = float(e)
                            if abs(e - v) <= 1e-12 * max(1.0, abs(e)):
                                values.at[i, c] = e
                        elif isinstance(e, float) and pd.notna(e) and pd.notna(v) and abs(e - v) <= 1e-12 * max(1.0, abs(e)):
                            values.at[i, c] = e
            ds.values = values.reset_index(drop=True)
            ds.next_id = int(max(values[ROW_ID].max() + 1, len(ds.original)))
            ds.journal = book['Журнал'].fillna('').astype(str).to_dict(orient='records')
            ds.undo_stack = []
            ds.version += 1
            return ds
        orig_ids = {A.probe_key(n): rid for n, rid in zip(ds.original[NAME], ds.original[ROW_ID])}
        ids, nxt = [], len(ds.original)
        for n, added in zip(values[NAME], values[ADDED]):
            key = A.probe_key(n)
            if not added and key in orig_ids:
                ids.append(orig_ids[key])
            else:
                ids.append(nxt)
                nxt += 1
        values[ROW_ID] = ids
        ds.values = values.reset_index(drop=True)
        ds.next_id = nxt
        ds.journal = book['Журнал'].fillna('').astype(str).to_dict(orient='records')
        ds.undo_stack = []
        ds.version += 1
        return ds
