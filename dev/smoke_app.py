"""Прогон всех разделов сайта через streamlit AppTest (без браузера)."""
import sys, time
from streamlit.testing.v1 import AppTest
pages = ['📋 Данные', '🧩 Кластеризация', '📈 Корреляции', '✅ Валидация']
at = AppTest.from_file('../app.py', default_timeout=900)
t0 = time.time(); at.run(); print('initial', round(time.time()-t0,1), 's; exceptions:', [e.value for e in at.exception])
for p in pages[1:] + pages[:1]:
    t0 = time.time()
    at.segmented_control(key='page').set_value(p).run()
    print(p, round(time.time()-t0,1), 's; exceptions:', [e.value for e in at.exception][:3],
          '; errors:', [e.value[:100] for e in at.error][:3])
