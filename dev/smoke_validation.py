"""Нажимает «Рассчитать» на вкладке «Валидация» с малым числом перестановок (AppTest)."""
import time
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('../app.py', default_timeout=1200)
at.run()
at.segmented_control(key='page').set_value('✅ Валидация').run()
at.slider(key='n_cond').set_value(100)
at.slider(key='n_full').set_value(50)
at.slider(key='n_yr').set_value(50)
at.run()
t0 = time.time()
[b for b in at.button if 'Рассчитать' in b.label][0].click().run()
print('run', round(time.time() - t0, 1), 's; exceptions:', [e.value for e in at.exception])
print('errors:', [e.value[:120] for e in at.error])
print('captions with p:', [c.value[:150] for c in at.caption if 'Q²' in c.value or 'Оценка' in c.value])
v = at.session_state['validation']
print({d: round(r['p'], 4) for d, r in v['cond'].items()})
print({d: round(r['p'], 4) for d, r in v['full'].items()})
print('y-rand p', v['yr']['p'] if v['yr'] else None)
