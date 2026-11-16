"""English site: all pages under several settings + manual k + validation run (AppTest)."""
from streamlit.testing.v1 import AppTest
PAGES = ['📋 Data', '🧩 Clustering', '📈 Correlations', '✅ Validation']
at = AppTest.from_file('../app_en.py', default_timeout=900)
at.run()
at.run()
def sweep(label):
    bad = []
    for p in PAGES:
        at.segmented_control(key='page').set_value(p).run()
        bad += [f'{p}: {e.value}' for e in at.exception]
    print(f'{label}: ' + ('OK' if not bad else ' | '.join(bad)[:800]))
sweep('default')
at.radio(key='units').set_value('g/g').run(); sweep('g/g')
at.radio(key='units').set_value('mmol/g').run()
at.radio(key='tau_mode').set_value('Fixed').run(); sweep('fixed tau')
at.radio(key='tau_mode').set_value('Auto (grid, as in notebook)').run()
at.segmented_control(key='page').set_value('🧩 Clustering').run()
at.toggle(key='km_auto').set_value(False).run(); at.slider(key='km_k').set_value(4).run()
print('manual k:', [e.value for e in at.exception] or 'OK', [w.value[:90] for w in at.warning][:1])
at.segmented_control(key='page').set_value('✅ Validation').run()
at.slider(key='n_cond').set_value(100); at.slider(key='n_full').set_value(50); at.slider(key='n_yr').set_value(50); at.run()
[b for b in at.button if 'Run' in b.label][0].click().run()
print('validation run:', [e.value for e in at.exception] or 'OK')
v = at.session_state['validation']; print({d: round(r['p'], 4) for d, r in list(v['cond'].items())[:4]})
texts = ' '.join([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info])
import re
print('cyrillic in rendered text:', bool(re.search('[А-Яа-яЁё]', texts)))
