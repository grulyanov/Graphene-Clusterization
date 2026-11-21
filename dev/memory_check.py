"""Peak memory of the full permutation test for different numbers of worker processes.

Run from the project root:  python dev/memory_check.py
"""
import os, sys, threading, time, subprocess
sys.path.insert(0, os.getcwd())
import analysis as A, data as D
ds = D.Dataset(D.read_excel_raw()); ds.preset('notebook'); f = ds.analysis_frame()
r = A.analyze_descriptor(f, 'β', A.SORPTION_MMOL, require_positive=True)
peak = {'main': 0, 'workers': 0, 'n': 0}
def watch():
    me = os.getpid()
    while not stop:
        out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,rss='], capture_output=True, text=True).stdout.split('\n')
        rows = [l.split() for l in out if l.strip()]
        main = sum(int(x[2]) for x in rows if int(x[0]) == me)
        kids = [int(x[2]) for x in rows if int(x[1]) == me]
        peak['main'] = max(peak['main'], main); peak['workers'] = max(peak['workers'], sum(kids)); peak['n'] = max(peak['n'], len(kids))
        time.sleep(0.3)
for nj in (1, 2, 16):
    stop = False; peak.update(main=0, workers=0, n=0)
    th = threading.Thread(target=watch); th.start()
    st = dict(n_iterations=5, min_r2=0.5, min_cv_r2=0.5, require_positive=True, tau_grid=None, threshold=None)
    A.permutation_test(f, 'β', r['models'], n_permutations=200, mode='full', settings=st, n_jobs=nj)
    time.sleep(1); stop = True; th.join()
    print(f'n_jobs={nj}: main {peak["main"]/1024:.0f} MB, workers {peak["n"]} × total {peak["workers"]/1024:.0f} MB')
    from joblib.externals.loky import get_reusable_executor; get_reusable_executor().shutdown(wait=True)
