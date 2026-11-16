"""Generate app_en.py (English site) from app.py (Russian site).

    python dev/make_en.py

Every Cyrillic string token of app.py is replaced by its translation from
dev/translations_en.py; Russian comments are dropped; the English text module,
language switch and dark-blue theme are inserted. The script stops and lists any
untranslated strings, so after editing app.py run it again and add the new keys.
"""
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dev'))
from translations_en import TR  # noqa: E402

CYR = re.compile('[А-Яа-яЁё]')

HEADER_CODE = '''import streamlit as st
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
'''

EXTRA_CSS = '''
h1, h2, h3, h4 {color: #0f2747;}
h1 {font-weight: 800; letter-spacing: -0.5px;}
.headline {background: #edf2f9 !important; border-left: 4px solid #1d4e89; border-radius: 4px !important;}
[data-testid="stSidebar"] h2 {color: #ffffff;}
[data-testid="stExpander"] details {border-color: #cfd9e8;}
</style>'''


def main():
    src = (ROOT / 'app.py').read_text(encoding='utf-8')
    line_starts = [0]
    for line in src.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def off(pos):
        return line_starts[pos[0] - 1] + pos[1]

    reps, missing = [], []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and CYR.search(tok.string):
            reps.append((off(tok.start), off(tok.end), ''))
        elif tok.type in (tokenize.STRING, tokenize.FSTRING_MIDDLE) and CYR.search(tok.string):
            if tok.string not in TR:
                missing.append(tok.string)
                continue
            a, b = off(tok.start), off(tok.end)
            if src[a:b] != tok.string:          # tokenizer position quirk: locate exactly
                a = src.find(tok.string, a - 2)
                b = a + len(tok.string)
            assert src[a:b] == tok.string, tok.string
            reps.append((a, b, TR[tok.string]))
    if missing:
        print('Untranslated strings (add them to dev/translations_en.py):')
        for m in dict.fromkeys(missing):
            print('   ', repr(m))
        sys.exit(1)
    out = src
    for a, b, new in sorted(reps, reverse=True):
        out = out[:a] + new + out[b:]
    out = re.sub(r'[ \t]+\n', '\n', out)                       # trailing spaces left by removed comments
    out = out.replace('«', '“').replace('»', '”')
    out = out.replace('from explain import T', 'from explain_en import T')
    assert out.count('import streamlit as st\n') == 1
    out = out.replace('import streamlit as st\n', HEADER_CODE, 1)
    assert out.count('</style>') == 1
    out = out.replace('</style>', EXTRA_CSS, 1)
    left = [ln for ln in out.splitlines() if CYR.search(ln)]
    if left:
        print('Cyrillic left in app_en.py:')
        print('\n'.join(left[:20]))
        sys.exit(1)
    (ROOT / 'app_en.py').write_text(out, encoding='utf-8')
    print(f'app_en.py written: {len(reps)} replacements')


if __name__ == '__main__':
    main()
