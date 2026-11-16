"""Язык интерфейса. Русский сайт — app.py (LANG = 'ru'), английский — app_en.py,
который ставит LANG = 'en' до импорта data и analysis."""
LANG = 'ru'


def t(ru: str, en: str) -> str:
    return en if LANG == 'en' else ru
