"""Ручное k: выключаем «оптимальное k», ставим k = 4 у K-Means и иерархической."""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('../app.py', default_timeout=900)
at.run()
at.segmented_control(key='page').set_value('🧩 Кластеризация').run()
print('auto:', [s.value[:110] for s in at.success][:3])
at.toggle(key='km_auto').set_value(False).run()
at.slider(key='km_k').set_value(4).run()
at.toggle(key='hc_auto').set_value(False).run()
at.slider(key='hc_k').set_value(3).run()
at.selectbox(key='hc_link').set_value('average').run()
print('exceptions:', [e.value for e in at.exception])
print('warnings:', [w.value[:160] for w in at.warning][:4])
print('captions:', [c.value[:90] for c in at.caption if 'Графиков' in c.value])
