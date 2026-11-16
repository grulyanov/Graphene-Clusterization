"""Список всех строковых фрагментов с кириллицей в файле (для словаря переводов)."""
import io, re, sys, tokenize
CYR = re.compile('[А-Яа-яЁё]')
src = open(sys.argv[1], encoding='utf-8').read()
seen = []
for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type in (tokenize.STRING, getattr(tokenize, 'FSTRING_MIDDLE', -1)) and CYR.search(tok.string):
        if tok.string not in seen:
            seen.append(tok.string)
for s in seen:
    print(repr(s))
print(len(seen), file=sys.stderr)
