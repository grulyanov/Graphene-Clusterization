"""Широкий прогон: все вкладки при разных настройках (AppTest)."""
from streamlit.testing.v1 import AppTest
PAGES = ['📋 Данные', '🧩 Кластеризация', '📈 Корреляции', '✅ Валидация']
at = AppTest.from_file('../app.py', default_timeout=900)
at.run()
def sweep(label):
    bad = []
    for p in PAGES:
        at.segmented_control(key='page').set_value(p).run()
        bad += [f'{p}: {e.value}' for e in at.exception]
    print(f'{label}: ' + ('OK' if not bad else ' | '.join(bad)[:600]))
sweep('по умолчанию')
at.radio(key='units').set_value('g/g').run(); sweep('г/г')
at.radio(key='units').set_value('mmol/g').run()
at.radio(key='tau_mode').set_value('Фиксированный').run(); sweep('фиксированный τ')
at.radio(key='tau_mode').set_value('Автоматически (сетка, как в ноутбуке)').run()
at.number_input(key='seed').set_value(7).run(); sweep('seed=7')
at.segmented_control(key='page').set_value('🧩 Кластеризация').run()
at.toggle(key='db_auto').set_value(False).run()
at.select_slider(key='db_eps').set_value(0.3).run()
at.toggle(key='sp_auto').set_value(False).run(); at.slider(key='sp_k').set_value(5).run()
print('ручные DBSCAN/спектральная:', [e.value for e in at.exception] or 'OK')
at.selectbox(key='clust_feature').set_value('β').run()
print('кластеризация по β:', [e.value for e in at.exception] or 'OK')
