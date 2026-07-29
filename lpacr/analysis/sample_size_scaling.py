"""Refinement: the Gaussian method-of-moments sigma_signal over-predicts the observed
z-distribution at k=1 (heavy tails). Fit a spike-and-slab marginal model instead, which
reproduces k=1 by construction, and redo the k-extrapolation. Also robust/trimmed MoM.
"""
import numpy as np
from scipy import stats, optimize

RNG = np.random.default_rng(11)
OUT = '/tmp/claude-1000/-home-emin-code-online-estimation/48a4ce1d-0725-4a5a-892e-9314841ef7ff/scratchpad'
D = dict(np.load(OUT + '/core_table.npz'))
D['elig'] = D['elig'].astype(bool)
V = D['valid'].astype(bool)
EARLY = D['early'].astype(bool); LATE = D['late'].astype(bool)
NULL1, NULL2 = 2 * (1 - stats.norm.cdf(1)), 2 * (1 - stats.norm.cdf(2))
EDGES = [16, 25, 40, 60]; LABELS = ['N<16', '16-25', '25-40', '40-60', '60+']
si = np.digitize(D['N_harm'], EDGES)


def nll(theta, lp, se):
    logitw, logtau = theta
    w = 1 / (1 + np.exp(-logitw)); tau2 = np.exp(2 * logtau)
    a = (1 - w) * stats.norm.pdf(lp, 0, se)
    b = w * stats.norm.pdf(lp, 0, np.sqrt(se ** 2 + tau2))
    return -np.sum(np.log(np.maximum(a + b, 1e-300)))


def fit_ss(lp, se):
    best = None
    for w0 in [-3.0, -1.0, 0.0, 2.0]:
        for t0 in [np.log(np.std(lp) + 1e-6), np.log(np.std(lp) * 3 + 1e-6), -2.0]:
            r = optimize.minimize(nll, [w0, t0], args=(lp, se), method='Nelder-Mead',
                                  options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    w = 1 / (1 + np.exp(-best.x[0])); tau = np.exp(best.x[1])
    # LR test vs pure noise (w=0)
    nll0 = -np.sum(np.log(np.maximum(stats.norm.pdf(lp, 0, se), 1e-300)))
    lr = 2 * (nll0 - best.fun)
    return w, tau, lr


def mc_ss(lp, se, w, tau, k, b, ndraw=4000):
    se_new = se * k ** b
    tau2 = tau ** 2
    d0 = stats.norm.pdf(lp, 0, se) * (1 - w)
    d1 = stats.norm.pdf(lp, 0, np.sqrt(se ** 2 + tau2)) * w
    post = d1 / np.maximum(d0 + d1, 1e-300)
    shr = tau2 / (tau2 + se ** 2)
    n = len(lp)
    isslab = RNG.random((ndraw, n)) < post
    true = np.where(isslab, shr * lp + np.sqrt(shr * se ** 2) * RNG.normal(size=(ndraw, n)), 0.0)
    z = np.abs(true + se_new * RNG.normal(size=(ndraw, n))) / se_new
    return (z > 1).mean(), (z > 2).mean()


def mom(lp, se, trim=0.0):
    if trim > 0:
        r = np.abs(lp) / se
        keep = r <= np.percentile(r, 100 * (1 - trim))
        lp, se = lp[keep], se[keep]
    return max(np.mean(lp ** 2) - np.mean(se ** 2), 0.0)


def mc_gauss(lp, se, s2, k, b, ndraw=4000):
    se_new = se * k ** b
    if s2 <= 0:
        true = np.zeros((ndraw, len(lp)))
    else:
        w = s2 / (s2 + se ** 2)
        true = w * lp + np.sqrt(w * se ** 2) * RNG.normal(size=(ndraw, len(lp)))
    z = np.abs(true + se_new * RNG.normal(size=(ndraw, len(lp)))) / se_new
    return (z > 1).mean(), (z > 2).mean()


BFIT = {'POOLED': -0.342, 'EARLY': -0.220, 'LATE': -0.476}

print('SPIKE-AND-SLAB model:  LP_i ~ (1-w) N(0,se_i^2) + w N(0, se_i^2 + tau^2)')
print('  w = fraction of cell-stages with real signal; tau = sd of true LP among those.')
print('  NULL is w=0 -> z half-normal for every k: P(z>1)=0.3173 P(z>2)=0.0455')
print('  LR stat vs w=0: ~chi2_1 mixture, 5% crit ~2.71 / 1% ~5.41\n')

for reg, mreg in [('POOLED', V), ('EARLY', V & EARLY), ('LATE', V & LATE)]:
    bfit = BFIT[reg]
    print('=' * 108)
    print(f'##### {reg}   (fitted SE exponent b={bfit:+.3f}; theory -0.500)')
    groups = [(l, mreg & (si == j)) for j, l in enumerate(LABELS)] + [('ALL', mreg),
                                                                     ('ALL-elig', mreg & D['elig'])]
    for lab, m in groups:
        if m.sum() < 30:
            print(f'{lab:9s} n={m.sum():4d}  too thin'); continue
        lp, se = D['lp'][m], D['lp_se'][m]
        w, tau, lr = fit_ss(lp, se)
        e1, e2 = (D['z'][m] > 1).mean(), (D['z'][m] > 2).mean()
        s2_mom = mom(lp, se); s2_tr = mom(lp, se, trim=0.05)
        print(f'\n{lab:9s} n={m.sum():4d} Nharm_med={np.median(D["N_harm"][m]):5.1f} | '
              f'spike-slab: w={w:.3f} tau={tau:.3f} (LR={lr:.1f})  | '
              f'sigma_MoM={np.sqrt(s2_mom):.3f}  sigma_MoM_trim5%={np.sqrt(s2_tr):.3f}')
        print(f'          observed k=1: P(z>1)={e1:.3f} (null .317, excess {e1-NULL1:+.3f})  '
              f'P(z>2)={e2:.3f} (null .045, excess {e2-NULL2:+.3f})')
        for mname, fn, par in [('spike-slab', lambda k, b: mc_ss(lp, se, w, tau, k, b), None),
                               ('gauss-MoM ', lambda k, b: mc_gauss(lp, se, s2_mom, k, b), None),
                               ('gauss-trim', lambda k, b: mc_gauss(lp, se, s2_tr, k, b), None)]:
            for bn, b in [('b=-0.500', -0.5), (f'b={bfit:+.3f}', bfit)]:
                res = [fn(k, b) for k in (1, 2, 4, 8)]
                s = '  '.join(f'k={k}:{p1:.3f}/{p2:.3f}' for k, (p1, p2) in zip((1, 2, 4, 8), res))
                ex = '  '.join(f'{p1-NULL1:+.3f}/{p2-NULL2:+.3f}' for (p1, p2) in res)
                print(f'          {mname} {bn} : {s}   | excess: {ex}')

# ---------------- LP autocorrelation, cross-cell within stage-pair
print('\n' + '=' * 108)
print('LP autocorrelation cross-check (background: median corr(LP_t,LP_t+1) = -0.446)')
stages = np.unique(D['stage']).astype(int)
rs_p, rs_s, st = [], [], []
for s in stages[:-1]:
    m0 = V & (D['stage'] == s); m1 = V & (D['stage'] == s + 1)
    c0, c1 = D['cell'][m0], D['cell'][m1]
    common = np.intersect1d(c0, c1)
    if len(common) < 20:
        continue
    a = D['lp'][m0][np.isin(c0, common)][np.argsort(c0[np.isin(c0, common)])]
    b = D['lp'][m1][np.isin(c1, common)][np.argsort(c1[np.isin(c1, common)])]
    rs_p.append(stats.pearsonr(a, b)[0]); rs_s.append(stats.spearmanr(a, b)[0]); st.append(s + 1)
rs_p, rs_s, st = np.array(rs_p), np.array(rs_s), np.array(st)
print(f'  cross-cell, per stage-pair: pearson median={np.median(rs_p):+.3f} '
      f'IQR[{np.percentile(rs_p,25):+.3f},{np.percentile(rs_p,75):+.3f}]  n_pairs={len(rs_p)}')
print(f'                              spearman median={np.median(rs_s):+.3f}')
print(f'    early(t+1<=16) pearson median={np.median(rs_p[st<=16]):+.3f} ; '
      f'late pearson median={np.median(rs_p[st>=17]):+.3f}')
print('    (pure-noise bound -0.500 ; independent-signal 0.000)')
# implied noise share from autocorr: rho = -sigma_noise^2/(sigma_noise^2+... ) heuristic
r = np.median(rs_p[st >= 17])
print(f'    implied noise variance share of LP in LATE from rho={r:+.3f}: '
      f'{min(-2*r,1.0)*100:.0f}% (rho=-0.5 <-> 100% noise, assuming iid perf noise & AR-free true perf)')

# ---------------- how much of LP_SE^2 is removable
print('\n' + '=' * 108)
print('Removable vs irreducible part of LP_SE (from sem^2 = A/N + C fit, LATE vs EARLY)')
for lab, mk in [('EARLY', EARLY), ('LATE', LATE)]:
    m = V & mk & np.isfinite(D['sem']) & (D['N'] >= 2)
    X = np.column_stack([1.0 / D['N'][m], np.ones(m.sum())])
    (A, C), *_ = np.linalg.lstsq(X, D['sem'][m] ** 2, rcond=None)
    for k in [1, 2, 4, 8]:
        Nk = np.median(D['N'][m]) * k
        print(f'  {lab} k={k}: N={Nk:5.0f} predicted sem={np.sqrt(max(A/Nk+C,0)):.4f} '
              f'(vs k^-0.5 naive {np.median(D["sem"][m])*k**-0.5:.4f}; observed median at k=1 '
              f'{np.median(D["sem"][m]):.4f})')
