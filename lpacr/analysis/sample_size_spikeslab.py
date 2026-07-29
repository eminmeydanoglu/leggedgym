"""Is the shallow cross-sectional b a real variance floor, or selection
(cells that get more episodes are cells with larger per-episode return SD)?

Cross-sectional b conflates the mechanical N^-1/2 with cell-level differences in
per-episode SD. Use two-way (cell + stage) fixed effects, which identifies b from
WITHIN-CELL changes in N over stages. Also inspect sd_episode = sem*sqrt(N) directly.
"""
import numpy as np
from scipy import stats

OUT = '/tmp/claude-1000/-home-emin-code-online-estimation/48a4ce1d-0725-4a5a-892e-9314841ef7ff/scratchpad'
D = dict(np.load(OUT + '/core_table.npz'))
V = D['valid'].astype(bool); EARLY = D['early'].astype(bool); LATE = D['late'].astype(bool)
SM = np.isfinite(D['sem']) & (D['sem'] > 0) & np.isfinite(D['N']) & (D['N'] >= 2)


def demean(v, g):
    out = v.copy()
    for gg in np.unique(g):
        m = g == gg
        out[m] -= out[m].mean()
    return out


def twoway(mask, label, iters=60):
    m = SM & mask
    y = np.log(D['sem'][m]); x = np.log(D['N'][m])
    c = D['cell'][m]; s = D['stage'][m]
    yy, xx = y.copy(), x.copy()
    for _ in range(iters):
        yy = demean(demean(yy, c), s); xx = demean(demean(xx, c), s)
    b = np.sum(xx * yy) / np.sum(xx * xx)
    r = yy - b * xx
    # cluster by cell
    XtXi = 1.0 / np.sum(xx * xx)
    meat = 0.0
    for cc in np.unique(c):
        mm = c == cc
        meat += (np.sum(xx[mm] * r[mm])) ** 2
    ng = len(np.unique(c))
    V_ = XtXi * meat * XtXi * ng / (ng - 1)
    se = np.sqrt(V_)
    # cell-FE only
    yy2, xx2 = demean(y, c), demean(x, c)
    b2 = np.sum(xx2 * yy2) / np.sum(xx2 * xx2)
    print(f'{label:26s} n={m.sum():5d} cells={ng:3d}  b_2wayFE={b:+.3f} 95%CI '
          f'[{b-1.96*se:+.3f},{b+1.96*se:+.3f}] (cluster by cell)   b_cellFEonly={b2:+.3f}   '
          f'theory -0.500')
    return b, se


print('TWO-WAY FIXED-EFFECT ESTIMATE OF b  (identifies from within-cell changes in N)')
bp = twoway(np.ones(len(D['sem']), bool), 'POOLED')
be = twoway(EARLY, 'EARLY 2-16')
bl = twoway(LATE, 'LATE 17-41')

print('\nPer-episode return SD  sd_ep = sem*sqrt(N).  If b==-0.5 exactly, sd_ep should be')
print('independent of N (null: spearman rho = 0).  rho>0 => high-N cells are also high-variance cells.')
for lab, mk in [('POOLED', np.ones(len(D['sem']), bool)), ('EARLY', EARLY), ('LATE', LATE)]:
    m = SM & mk
    sd = D['sem'][m] * np.sqrt(D['N'][m])
    r, p = stats.spearmanr(D['N'][m], sd)
    print(f'  {lab:7s} n={m.sum():5d} sd_ep med={np.median(sd):.3f} '
          f'IQR[{np.percentile(sd,25):.3f},{np.percentile(sd,75):.3f}]  '
          f'spearman(N, sd_ep)={r:+.3f} (p={p:.2g}, null 0.000)')
    # within-stage
    rs = []
    for s in np.unique(D['stage'][m]):
        mm = m & (D['stage'] == s)
        if mm.sum() > 20 and np.std(D['N'][mm]) > 0:
            rr, _ = stats.spearmanr(D['N'][mm], D['sem'][mm] * np.sqrt(D['N'][mm]))
            if np.isfinite(rr):
                rs.append(rr)
    print(f'          within-stage spearman(N, sd_ep): median={np.median(rs):+.3f} '
          f'IQR[{np.percentile(rs,25):+.3f},{np.percentile(rs,75):+.3f}] over {len(rs)} stages')
    # within-cell over stages
    rc = []
    for c in np.unique(D['cell'][m]):
        mm = m & (D['cell'] == c)
        if mm.sum() > 10 and np.std(D['N'][mm]) > 0:
            rr, _ = stats.spearmanr(D['N'][mm], D['sem'][mm] * np.sqrt(D['N'][mm]))
            if np.isfinite(rr):
                rc.append(rr)
    print(f'          within-cell  spearman(N, sd_ep): median={np.median(rc):+.3f} '
          f'IQR[{np.percentile(rc,25):+.3f},{np.percentile(rc,75):+.3f}] over {len(rc)} cells')

print('\nCross-sectional (stage-FE) b vs two-way b:')
print('  POOLED  cross=-0.342  twoway=%+.3f' % bp[0])
print('  EARLY   cross=-0.220  twoway=%+.3f' % be[0])
print('  LATE    cross=-0.476  twoway=%+.3f' % bl[0])

# Decision numbers: SE reduction factor at k
print('\nSE reduction factor at k x budget, per estimate of b (LATE regime is the operating point):')
for lab, b in [('theory', -0.5), ('LATE cross-sec', -0.476), ('LATE two-way', bl[0]),
               ('EARLY cross-sec', -0.220), ('EARLY two-way', be[0])]:
    print(f'  {lab:16s} b={b:+.3f}: k=2 x{2**b:.3f}  k=4 x{4**b:.3f}  k=8 x{8**b:.3f}')

# how much N to halve LP_SE
print('\nEpisodes needed to halve LP_SE (from median admitted N=42):')
for lab, b in [('theory -0.5', -0.5), ('LATE two-way', bl[0]), ('EARLY two-way', be[0])]:
    k = 0.5 ** (1.0 / b)
    print(f'  {lab:16s} k={k:.2f}x  -> N={42*k:.0f} admitted episodes/cell/stage')
