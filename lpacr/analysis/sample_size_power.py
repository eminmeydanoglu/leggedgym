"""LP noise / sample-size analysis for lpacrl-fixedbeta-seed1.

Offline analysis only. Outputs a text report to stdout and a .npz of the core table.
"""
import json, sys
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260728)
PATH = ('/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/logs/'
        'curriculum_atlas_local/505449_20260727_155936/lpacrl-fixedbeta-seed1/frames.ndjson')
OUT = '/tmp/claude-1000/-home-emin-code-online-estimation/48a4ce1d-0725-4a5a-892e-9314841ef7ff/scratchpad'

F = [json.loads(l) for l in open(PATH)]
NF = len(F)
NC = 84


def arr(i, k):
    return np.asarray(F[i]['metrics'][k], dtype=float)


def barr(i, k):
    return np.asarray(F[i]['metrics'][k], dtype=bool)


# ---------------------------------------------------------------- core table
rows = []
for i in range(1, NF):                      # frame idx 1..40  -> stage_index 2..41
    stage = F[i]['metadata']['frame']['stage_index']
    perf, perf_p = arr(i, 'performance'), arr(i - 1, 'performance')
    sem, sem_p = arr(i, 'performance_sem'), arr(i - 1, 'performance_sem')
    lp_file = arr(i, 'learning_progress')
    elp = arr(i, 'effective_learning_progress')
    n, n_p = arr(i, 'stage_episode_count'), arr(i, 'previous_stage_episode_count')
    n_prev_frame = arr(i - 1, 'stage_episode_count')
    elig = barr(i, 'eligible_for_lp')
    sp = arr(i, 'sampling_probability')
    sp_prev = arr(i - 1, 'sampling_probability')
    lp = perf - perf_p
    lp_se = np.sqrt(sem ** 2 + sem_p ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        n_harm = 2.0 / (1.0 / n + 1.0 / n_p)
    for c in range(NC):
        rows.append((stage, c, perf[c], perf_p[c], sem[c], sem_p[c], lp[c], lp_file[c],
                     elp[c], n[c], n_p[c], n_prev_frame[c], n_harm[c], elig[c], sp[c], sp_prev[c]))

names = ['stage', 'cell', 'perf', 'perf_p', 'sem', 'sem_p', 'lp', 'lp_file', 'elp',
         'N', 'Np', 'N_prevframe', 'N_harm', 'elig', 'sp', 'sp_prev']
D = {k: np.array([r[j] for r in rows], float) for j, k in enumerate(names)}
D['elig'] = D['elig'].astype(bool)
D['lp_se'] = np.sqrt(D['sem'] ** 2 + D['sem_p'] ** 2)
D['z'] = np.abs(D['lp']) / D['lp_se']

# sanity: previous_stage_episode_count vs stage_episode_count of previous frame
ok = np.isfinite(D['Np']) & np.isfinite(D['N_prevframe'])
print('# sanity: Np == N of prev frame for %.3f of rows' %
      np.mean(D['Np'][ok] == D['N_prevframe'][ok]))
m = np.isfinite(D['lp']) & np.isfinite(D['lp_file'])
print('# sanity: |lp_computed - lp_file| max = %.3g on %d rows (file LP finite on %d)'
      % (np.max(np.abs(D['lp'][m] - D['lp_file'][m])), m.sum(), np.isfinite(D['lp_file']).sum()))

VALID = np.isfinite(D['lp']) & np.isfinite(D['lp_se']) & (D['lp_se'] > 0) & np.isfinite(D['N_harm'])
D['valid'] = VALID
EARLY = (D['stage'] >= 2) & (D['stage'] <= 16)
LATE = (D['stage'] >= 17)
D['early'], D['late'] = EARLY, LATE
print('# rows total %d ; valid (LP & LP_SE defined) %d ; early %d ; late %d'
      % (len(rows), VALID.sum(), (VALID & EARLY).sum(), (VALID & LATE).sum()))
print('# eligible_for_lp among valid: %d (%.1f%%)' % ((VALID & D['elig']).sum(),
                                                      100 * (VALID & D['elig']).mean()))

np.savez(OUT + '/core_table.npz', **{k: v for k, v in D.items()})


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan,) * 3
    return np.percentile(x, 25), np.median(x), np.percentile(x, 75)


def fmt(x, p=3):
    return 'nan' if not np.isfinite(x) else f'{x:.{p}g}'


def miqr(x, p=3):
    a, b, c = q(x)
    return f'{fmt(b,p)} [{fmt(a,p)},{fmt(c,p)}]'


# ================================================== TASK 2 : strata
print('\n' + '=' * 100)
print('TASK 2 - distributions by N_harm stratum')
EDGES = [0, 16, 25, 40, 60, np.inf]
LABELS = ['N<16', '16-25', '25-40', '40-60', '60+']


def strat_idx(nh):
    return np.digitize(nh, EDGES[1:-1], right=False)


def table2(mask, title):
    print(f'\n-- {title} (n={mask.sum()}) --')
    print(f'{"stratum":9s} {"n":>5s} {"N_harm med":>12s} {"LP_SE med[IQR]":>26s} '
          f'{"|LP| med[IQR]":>26s} {"z med[IQR]":>22s} {"samp_prob med[IQR]":>26s} {"sem med[IQR]":>26s}')
    si = strat_idx(D['N_harm'])
    for j, lab in enumerate(LABELS):
        m = mask & (si == j)
        if m.sum() == 0:
            print(f'{lab:9s} {0:5d}   (empty)')
            continue
        print(f'{lab:9s} {m.sum():5d} {fmt(np.median(D["N_harm"][m])):>12s} '
              f'{miqr(D["lp_se"][m]):>26s} {miqr(np.abs(D["lp"][m])):>26s} '
              f'{miqr(D["z"][m]):>22s} {miqr(D["sp"][m]):>26s} {miqr(D["sem"][m]):>26s}')
    m = mask
    print(f'{"ALL":9s} {m.sum():5d} {fmt(np.median(D["N_harm"][m])):>12s} '
          f'{miqr(D["lp_se"][m]):>26s} {miqr(np.abs(D["lp"][m])):>26s} '
          f'{miqr(D["z"][m]):>22s} {miqr(D["sp"][m]):>26s} {miqr(D["sem"][m]):>26s}')
    # quintiles too
    nh = D['N_harm'][mask]
    qe = np.percentile(nh, [20, 40, 60, 80])
    print(f'   quintile edges of N_harm: {np.round(qe,1)}')
    qi = np.digitize(D['N_harm'], qe)
    for j in range(5):
        m2 = mask & (qi == j)
        print(f'   Q{j+1} n={m2.sum():4d} N_harm med={fmt(np.median(D["N_harm"][m2])):>7s} '
              f'z med={fmt(np.median(D["z"][m2])):>6s}  |LP| med={fmt(np.median(np.abs(D["lp"][m2]))):>8s} '
              f' LP_SE med={fmt(np.median(D["lp_se"][m2])):>8s}')


table2(VALID, 'POOLED stages 2-41')
table2(VALID & EARLY, 'EARLY stages 2-16')
table2(VALID & LATE, 'LATE stages 17-41')


# ================================================== TASK 3 : 1/sqrt(N) scaling
print('\n' + '=' * 100)
print('TASK 3 - does performance_sem scale as N^-1/2 ?')

SM = np.isfinite(D['sem']) & (D['sem'] > 0) & np.isfinite(D['N']) & (D['N'] >= 2)
y = np.log(D['sem'][SM]); x = np.log(D['N'][SM]); st = D['stage'][SM]


def ols_fe(x, y, groups):
    """OLS of y on x with group fixed effects; returns b, se_cluster, n."""
    g = np.unique(groups)
    X = np.zeros((len(x), 1 + len(g)))
    X[:, 0] = x
    for j, gg in enumerate(g):
        X[:, 1 + j] = (groups == gg)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gg in g:
        m = groups == gg
        u = X[m].T @ r[m]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    ng = len(g)
    V *= ng / max(ng - 1, 1)
    return beta[0], np.sqrt(V[0, 0]), len(x), ng


def report_fit(mask, title):
    m = SM & mask
    yy, xx, ss = np.log(D['sem'][m]), np.log(D['N'][m]), D['stage'][m]
    if m.sum() < 10:
        print(f'{title}: too few'); return
    b, se, n, ng = ols_fe(xx, yy, ss)
    lo, hi = b - 1.96 * se, b + 1.96 * se
    # simple pooled no-FE
    sl, ic, r_, p_, se_ = stats.linregress(xx, yy)
    print(f'{title:22s} n={n:5d} stages={ng:3d}  b_FE={b:+.3f}  95%CI [{lo:+.3f},{hi:+.3f}] '
          f' (cluster-by-stage)   b_noFE={sl:+.3f}  R2={r_**2:.3f}   dev_from_-0.5 = {b+0.5:+.3f} '
          f'({"SHALLOWER" if lo > -0.5 else "consistent with -0.5" if hi > -0.5 else "steeper"})')
    return b, lo, hi


bp = report_fit(np.ones(len(y := D['sem']), bool), 'POOLED (stage FE)')
be = report_fit(EARLY, 'EARLY 2-16 (stage FE)')
bl = report_fit(LATE, 'LATE 17-41 (stage FE)')

print('\nper-stage slopes (no FE):')
slopes = []
for s in np.unique(D['stage']):
    m = SM & (D['stage'] == s)
    if m.sum() < 8 or np.std(np.log(D['N'][m])) < 1e-6:
        continue
    sl, ic, r_, p_, se_ = stats.linregress(np.log(D['N'][m]), np.log(D['sem'][m]))
    slopes.append((int(s), m.sum(), sl, se_, r_ ** 2))
for s, n, sl, se_, r2 in slopes:
    print(f'  stage {s:3d}  n={n:3d}  b={sl:+.3f} +- {se_:.3f}   R2={r2:.2f}')
sv = np.array([s[2] for s in slopes]); sst = np.array([s[0] for s in slopes])
print(f'  median per-stage b : all={np.median(sv):+.3f}  early={np.median(sv[sst<=16]):+.3f} '
      f' late={np.median(sv[sst>=17]):+.3f}   (theory -0.500)')
print(f'  #stages with b > -0.4 : {(sv>-0.4).sum()}/{len(sv)}')

# variance decomposition implied by b
for lab, bb in [('pooled', bp[0]), ('early', be[0]), ('late', bl[0])]:
    print(f'  implied: at k x budget, SE multiplier k^{bb:+.3f} vs theory k^-0.500. '
          f'k=8 -> x{8**bb:.3f} (theory x{8**-0.5:.3f})   floor ratio')

# direct variance-floor model: sem^2 = A/N + C  (C = irreducible)
print('\n  direct fit sem^2 = A/N + C (per regime, OLS on 1/N):')
for lab, mk in [('pooled', np.ones(len(D['sem']), bool)), ('early', EARLY), ('late', LATE)]:
    m = SM & mk
    X = np.column_stack([1.0 / D['N'][m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(X, D['sem'][m] ** 2, rcond=None)
    A, C = coef
    # what N would be needed to reach half the current median sem?
    med_sem = np.median(D['sem'][m])
    floor = np.sqrt(max(C, 0))
    print(f'    {lab:6s} A={A:.4g} C={C:.4g} -> irreducible SE floor={floor:.4g}; '
          f'current median sem={med_sem:.4g}; floor/median={floor/med_sem:.2f} '
          f'(N->inf can only remove {(1-floor/med_sem)*100:.0f}% of SE)')

# ================================================== TASK 4 : extrapolation
print('\n' + '=' * 100)
print('TASK 4 - extrapolate to k x episode budget')
print('NULL (sigma_signal = 0): z ~ half-normal for ALL k -> P(z>1)=0.3173, P(z>2)=0.0455')

NULL1, NULL2 = 2 * (1 - stats.norm.cdf(1)), 2 * (1 - stats.norm.cdf(2))
NDRAW = 4000


def decompose(lp, se):
    s2 = np.mean(lp ** 2) - np.mean(se ** 2)
    return max(s2, 0.0), s2


def mc(lp, se, sigma2, k, b, mode='posterior', ndraw=NDRAW):
    """returns P(z>1), P(z>2) at k x budget."""
    se_new = se * (k ** b)
    n = len(lp)
    if sigma2 <= 0:
        true = np.zeros((ndraw, n))
    elif mode == 'prior':
        true = RNG.normal(0.0, np.sqrt(sigma2), size=(ndraw, n))
    else:  # empirical-Bayes posterior for each cell's own true LP
        w = sigma2 / (sigma2 + se ** 2)
        mu = w * lp
        sd = np.sqrt(w * se ** 2)
        true = mu + sd * RNG.normal(size=(ndraw, n))
    obs = true + se_new * RNG.normal(size=(ndraw, n))
    z = np.abs(obs) / se_new
    return (z > 1).mean(), (z > 2).mean()


def task4(mask, title, bfit):
    print(f'\n===== {title} =====')
    si = strat_idx(D['N_harm'])
    groups = [(lab, mask & (si == j)) for j, lab in enumerate(LABELS)] + [('ALL', mask)]
    for lab, m in groups:
        if m.sum() < 15:
            print(f'{lab:7s} n={m.sum():4d}  (too thin, skipped)')
            continue
        lp, se = D['lp'][m], D['lp_se'][m]
        s2, s2raw = decompose(lp, se)
        sig = np.sqrt(s2)
        emp1, emp2 = (D['z'][m] > 1).mean(), (D['z'][m] > 2).mean()
        snr = s2 / np.mean(se ** 2)
        print(f'\n{lab:7s} n={m.sum():4d}  N_harm med={np.median(D["N_harm"][m]):6.1f}  '
              f'mean(LP^2)={np.mean(lp**2):.4g} mean(LP_SE^2)={np.mean(se**2):.4g}  '
              f'sigma_signal^2={s2raw:+.4g}->{s2:.4g} (sigma={sig:.4g})  '
              f'var-ratio signal/noise={snr:+.3f}')
        print(f'        observed  P(z>1)={emp1:.3f} (null .317, excess {emp1-NULL1:+.3f})   '
              f'P(z>2)={emp2:.3f} (null .045, excess {emp2-NULL2:+.3f})')
        for bname, b in [('theory b=-0.500', -0.5), (f'fitted b={bfit:+.3f}', bfit)]:
            out = []
            for k in [1, 2, 4, 8]:
                p1, p2 = mc(lp, se, s2, k, b)
                out.append((k, p1, p2))
            s = '   '.join(f'k={k}: {p1:.3f}/{p2:.3f}' for k, p1, p2 in out)
            print(f'        MC[{bname}] P(z>1)/P(z>2) : {s}')
            e = '   '.join(f'k={k}: {p1-NULL1:+.3f}/{p2-NULL2:+.3f}' for k, p1, p2 in out)
            print(f'        excess over null                : {e}')
        # prior-draw variant for ALL only
        if lab == 'ALL':
            for b in [-0.5, bfit]:
                out = [mc(lp, se, s2, k, b, mode='prior') for k in [1, 2, 4, 8]]
                s = '   '.join(f'k={k}: {p1:.3f}/{p2:.3f}' for k, (p1, p2) in zip([1, 2, 4, 8], out))
                print(f'        [prior-draw variant, b={b:+.3f}] {s}')


task4(VALID, 'POOLED stages 2-41', bp[0])
task4(VALID & EARLY, 'EARLY stages 2-16', be[0])
task4(VALID & LATE, 'LATE stages 17-41', bl[0])

# also: eligible-only version (what the sampler actually uses)
print('\n----- eligible_for_lp == True only -----')
task4(VALID & D['elig'], 'ELIGIBLE pooled', bp[0])
task4(VALID & D['elig'] & LATE, 'ELIGIBLE late 17-41', bl[0])

# how many extra cells cross z>1 in absolute terms
print('\n-- absolute cell counts (LATE regime, eligible) --')
m = VALID & D['elig'] & LATE
lp, se = D['lp'][m], D['lp_se'][m]
s2, _ = decompose(lp, se)
per_stage = m.sum() / len(np.unique(D['stage'][m]))
for b in [-0.5, bl[0]]:
    for k in [1, 2, 4, 8]:
        p1, p2 = mc(lp, se, s2, k, b)
        print(f'   b={b:+.3f} k={k}: expected cells/stage with z>1 = {p1*per_stage:.1f} '
              f'(null {NULL1*per_stage:.1f}), z>2 = {p2*per_stage:.2f} (null {NULL2*per_stage:.2f}) '
              f'of {per_stage:.0f} eligible cells/stage')

# ================================================== TASK 5
print('\n' + '=' * 100)
print('TASK 5 - low-N cells: extreme LP and runaway risk')
print('NULL for all Spearman rho below: 0.000; |rho| that is 5% significant ~ 1.96/sqrt(n)')


def sp_corr(a, b, m, lab):
    mm = m & np.isfinite(a) & np.isfinite(b)
    if mm.sum() < 10:
        print(f'  {lab}: n={mm.sum()} too few'); return
    r, p = stats.spearmanr(a[mm], b[mm])
    rp, pp = stats.pearsonr(a[mm], b[mm])
    crit = 1.96 / np.sqrt(mm.sum())
    print(f'  {lab:44s} n={mm.sum():5d}  rho={r:+.3f} (p={p:.2g}, |rho|_crit={crit:.3f})  '
          f'[pearson {rp:+.3f}]')


for lab, m in [('POOLED', VALID), ('EARLY 2-16', VALID & EARLY), ('LATE 17-41', VALID & LATE)]:
    print(f'\n-- {lab} --')
    sp_corr(D['N_harm'], np.abs(D['lp']), m, 'N_harm vs |LP|')
    sp_corr(D['N_harm'], D['z'], m, 'N_harm vs z')
    sp_corr(D['N_harm'], D['sp'], m, 'N_harm vs sampling_probability')
    sp_corr(D['N_harm'], D['lp_se'], m, 'N_harm vs LP_SE')
    # within-stage (rank within each stage) to remove stage-level shifts
    for aname, a in [('|LP|', np.abs(D['lp'])), ('sampling_probability', D['sp'])]:
        rs, rn = [], []
        for s in np.unique(D['stage'][m]):
            mm = m & (D['stage'] == s) & np.isfinite(a) & np.isfinite(D['N_harm'])
            if mm.sum() < 10 or np.std(D['N_harm'][mm]) == 0:
                continue
            r, _ = stats.spearmanr(D['N_harm'][mm], a[mm])
            if np.isfinite(r):
                rs.append(r); rn.append(s)
        if rs:
            rs = np.array(rs)
            t = stats.wilcoxon(rs) if len(rs) > 5 else (np.nan, np.nan)
            print(f'  within-stage N_harm vs {aname:22s} median rho={np.median(rs):+.3f} '
                  f'over {len(rs)} stages (IQR {np.percentile(rs,25):+.3f},{np.percentile(rs,75):+.3f}) '
                  f'wilcoxon p={getattr(t,"pvalue",np.nan):.2g}')

print('\n-- top-10 cells by sampling_probability: fraction in bottom quartile of N_harm --')
print('   (null = 0.25 by construction)')
for lab, m in [('POOLED', VALID), ('EARLY 2-16', VALID & EARLY), ('LATE 17-41', VALID & LATE)]:
    fr = []
    for s in np.unique(D['stage'][m]):
        mm = m & (D['stage'] == s)
        if mm.sum() < 20:
            continue
        idx = np.where(mm)[0]
        nh = D['N_harm'][idx]; spp = D['sp'][idx]
        q1 = np.percentile(nh, 25)
        top = idx[np.argsort(-spp)[:10]]
        fr.append(np.mean(D['N_harm'][top] <= q1))
    fr = np.array(fr)
    print(f'  {lab:11s} stages={len(fr):3d}  mean fraction={np.mean(fr):.3f}  median={np.median(fr):.3f} '
          f' (null 0.250, excess {np.mean(fr)-0.25:+.3f})  stages with fraction>0.5: {(fr>0.5).sum()}')

print('\n-- gated-out (eligible_for_lp == False) vs eligible --')
for lab, m in [('POOLED', VALID), ('EARLY 2-16', VALID & EARLY), ('LATE 17-41', VALID & LATE)]:
    a = m & ~D['elig']; b = m & D['elig']
    print(f'  {lab}:  gated n={a.sum():4d}  eligible n={b.sum():4d}')
    for nm, v in [('N_harm', D['N_harm']), ('N (this stage)', D['N']), ('|LP|', np.abs(D['lp'])),
                  ('LP_SE', D['lp_se']), ('z', D['z']), ('sampling_prob', D['sp']),
                  ('|eff_LP|', np.abs(D['elp']))]:
        if a.sum() > 3 and b.sum() > 3:
            u = stats.mannwhitneyu(v[a][np.isfinite(v[a])], v[b][np.isfinite(v[b])])
            print(f'    {nm:16s} gated {miqr(v[a]):>26s}   eligible {miqr(v[b]):>26s}  MWU p={u.pvalue:.2g}')

print('\n-- feedback loop: sp at stage t  ->  N at stage t+1  ->  |LP| at t+1 --')
# build lag structure
stages = np.unique(D['stage']).astype(int)
sp_t, n_t1, lp_t1, z_t1, reg = [], [], [], [], []
for s in stages[:-1]:
    m0 = (D['stage'] == s); m1 = (D['stage'] == s + 1)
    if m0.sum() != NC or m1.sum() != NC:
        continue
    o0 = np.argsort(D['cell'][m0]); o1 = np.argsort(D['cell'][m1])
    sp_t.append(D['sp'][m0][o0]); n_t1.append(D['N'][m1][o1])
    lp_t1.append(np.abs(D['lp'][m1][o1])); z_t1.append(D['z'][m1][o1])
    reg.append(np.full(NC, s + 1))
sp_t = np.concatenate(sp_t); n_t1 = np.concatenate(n_t1)
lp_t1 = np.concatenate(lp_t1); z_t1 = np.concatenate(z_t1); reg = np.concatenate(reg)
for lab, m in [('POOLED', np.ones(len(sp_t), bool)), ('EARLY(t+1<=16)', reg <= 16),
               ('LATE(t+1>=17)', reg >= 17)]:
    sp_corr(sp_t, n_t1, m, f'{lab}: sp[t] vs N[t+1]')
    sp_corr(n_t1, lp_t1, m, f'{lab}: N[t+1] vs |LP|[t+1]')
    sp_corr(n_t1, z_t1, m, f'{lab}: N[t+1] vs z[t+1]')
    sp_corr(sp_t, lp_t1, m, f'{lab}: sp[t] vs |LP|[t+1]')

print('\n-- persistence of LP (background says median corr(LP_t,LP_t+1) = -0.446) --')
cors = []
for c in range(NC):
    m = VALID & (D['cell'] == c)
    o = np.argsort(D['stage'][m]); v = D['lp'][m][o]
    if len(v) > 8:
        r, _ = stats.spearmanr(v[:-1], v[1:])
        if np.isfinite(r):
            cors.append(r)
print(f'  per-cell spearman(LP_t, LP_t+1) over stages: median={np.median(cors):+.3f} '
      f'IQR[{np.percentile(cors,25):+.3f},{np.percentile(cors,75):+.3f}] n_cells={len(cors)} '
      f'(pure-noise bound -0.500, no-noise-signal 0.000)')

print('\n-- diagnostics from metadata --')
late = np.array([F[i]['metadata']['frame']['diagnostics']['late_outcome_count'] for i in range(NF)], float)
compl = np.array([np.nansum(arr(i, 'task_completion_count')) for i in range(NF)])
ess = np.array([F[i]['metadata']['frame']['diagnostics']['effective_sample_size'] for i in range(NF)], float)
lprel = np.array([F[i]['metadata']['frame']['diagnostics']['lp_reliability_median']
                  if F[i]['metadata']['frame']['diagnostics']['lp_reliability_median'] is not None else np.nan
                  for i in range(NF)], float)
dlate = np.diff(late); dcompl = np.diff(compl)
frac = dlate / np.maximum(dcompl, 1)
print(f'  per-stage late/completed fraction: median={np.nanmedian(frac):.3f} '
      f'IQR[{np.nanpercentile(frac,25):.3f},{np.nanpercentile(frac,75):.3f}]  '
      f'early={np.nanmedian(frac[:15]):.3f} late={np.nanmedian(frac[15:]):.3f}')
print(f'  ESS: early median={np.median(ess[1:16]):.1f} late median={np.median(ess[16:]):.1f} (max 84)')
print(f'  lp_reliability_median: early={np.nanmedian(lprel[1:16]):.3f} late={np.nanmedian(lprel[16:]):.3f}')
print(f'  admitted N per cell-stage: median={np.nanmedian(D["N"][VALID]):.1f} ; '
      f'implied uncensored ~{np.nanmedian(D["N"][VALID])/(1-np.nanmedian(frac)):.1f}')
