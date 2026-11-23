"""Streak detection on "new light" difference images.

new_light(t) = frame t - max(frame t-1, frame t+1), after scaling the
neighbours to frame t's sky level. Stars, the Moon, static clouds and
anything else present in a neighbour cancel; what's left is light that was
only there during exposure t: meteors, the part of a satellite or plane
track flown during t, and noise.

Thresholded pixels are grouped into connected fragments, fragments lying on
one line are merged (faint or dashed trails break up), and each merged
streak gets a brightness profile along its length, from which the scorer's
features are measured. Approach after allsky-meteordetect (MIT,
github.com/benhartwich/allsky-meteordetect): temporal differencing, then
continuation tests against the neighbouring frames.
"""

import math

import cv2
import numpy

K_SIGMA = 4.0           # detection threshold, x robust noise
MIN_FRAG_PX = 5         # fragment extent
MIN_STREAK_PX = 20      # merged streak length (~3.5 deg at the zenith)
MIN_ELONGATION = 5.0
MERGE_ANGLE_DEG = 6.0
MERGE_PERP_PX = 4.0
MERGE_GAP_PX = 60.0
MAX_COMPONENTS = 4000   # more = noise or cloud motion; skip the frame
BUSY_FRACTION = 0.02    # more of the sky above threshold = cloud/twilight


def gray(img):
    return cv2.GaussianBlur(img.astype(numpy.float32).mean(axis=2), (3, 3), 0)


def sky_level(g, mask_idx):
    return float(numpy.median(g.ravel()[mask_idx]))


def new_light(cur, refs, mask_idx):
    """cur - max(refs), each ref scaled to cur's median sky level. Returns
    (difference, scales) or (None, reason) when the exposure jumped."""
    level = sky_level(cur, mask_idx)
    best = None
    scales = []
    for r in refs:
        rl = sky_level(r, mask_idx)
        s = level / rl if rl > 0 else 1.0
        if not 0.5 < s < 2.0:
            return None, 'exposure jump'
        scales.append(s)
        rs = r * s
        best = rs if best is None else numpy.maximum(best, rs)
    return cur - best, scales


def robust_sigma(d, mask_idx):
    v = d.ravel()[mask_idx]
    med = numpy.median(v)
    return max(1.4826 * float(numpy.median(numpy.abs(v - med))), 0.5)


def _pca(xs, ys):
    pts = numpy.stack([xs, ys], axis=1).astype(numpy.float64)
    c = pts.mean(axis=0)
    cov = numpy.cov((pts - c).T) if len(pts) > 2 else numpy.eye(2)
    w, v = numpy.linalg.eigh(cov)
    d = v[:, 1]
    n = numpy.array([-d[1], d[0]])
    along = (pts - c) @ d
    perp = (pts - c) @ n
    return c, d, along, perp


def fragments(d, mask, sigma, k=K_SIGMA):
    """Line-ish connected components above k*sigma. Returns (list, busy_frac)
    or (None, reason)."""
    binary = ((d > k * sigma) & mask).astype(numpy.uint8)
    busy = float(binary.sum()) / max(int(mask.sum()), 1)
    if busy > BUSY_FRACTION:
        return None, f'busy {busy:.3f}'
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > MAX_COMPONENTS:
        return None, f'{n} components'
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4 or max(w, h) < MIN_FRAG_PX:
            continue
        ys, xs = numpy.nonzero(labels[y:y + h, x:x + w] == i)
        xs = xs + x
        ys = ys + y
        c, dvec, along, perp = _pca(xs, ys)
        length = float(along.max() - along.min() + 1)
        width = float(perp.max() - perp.min() + 1)
        if length / width < 2.0 and length < 12:
            continue  # blob: star residue, hot pixel, noise
        out.append({'c': c, 'd': dvec, 'len': length, 'xs': xs, 'ys': ys})
    return out, busy


def merge(frags):
    """Greedy merge of fragments on a common line into streaks."""
    frags = sorted(frags, key=lambda f: -f['len'])[:400]
    used = [False] * len(frags)
    cos_tol = math.cos(math.radians(MERGE_ANGLE_DEG))
    streaks = []
    for i, seed in enumerate(frags):
        if used[i]:
            continue
        used[i] = True
        xs, ys = [seed['xs']], [seed['ys']]
        c, d = seed['c'], seed['d']
        cx, cy = numpy.concatenate(xs), numpy.concatenate(ys)
        c, d, along, _ = _pca(cx, cy)
        lo, hi = along.min(), along.max()
        changed = True
        while changed:
            changed = False
            for j, f in enumerate(frags):
                if used[j] or abs(float(numpy.dot(f['d'], d))) < cos_tol:
                    continue
                off = f['c'] - c
                if abs(float(off[0] * -d[1] + off[1] * d[0])) > MERGE_PERP_PX + 0.02 * abs(float(off @ d)):
                    continue
                fa = numpy.stack([f['xs'] - c[0], f['ys'] - c[1]], axis=1) @ d
                gap = max(fa.min() - hi, lo - fa.max(), 0.0)
                if gap > MERGE_GAP_PX:
                    continue
                used[j] = True
                xs.append(f['xs'])
                ys.append(f['ys'])
                cx, cy = numpy.concatenate(xs), numpy.concatenate(ys)
                c, d, along, _ = _pca(cx, cy)
                lo, hi = along.min(), along.max()
                changed = True
        _, _, along, perp = _pca(cx, cy)
        length = float(hi - lo + 1)
        width = float(numpy.percentile(perp, 95) - numpy.percentile(perp, 5) + 1)
        p1 = c + d * lo
        p2 = c + d * hi
        streaks.append({'p1': p1, 'p2': p2, 'len': length, 'width': width, 'elong': length / max(width, 1.0),
                        'npix': int(len(cx)), 'nfrag': len(xs), 'xs': cx, 'ys': cy})
    return streaks


def detect(d, mask, sigma, min_len=MIN_STREAK_PX, min_elong=MIN_ELONGATION, k=K_SIGMA):
    frags, info = fragments(d, mask, sigma, k)
    if frags is None:
        return None, info
    streaks = [s for s in merge(frags) if s['len'] >= min_len and s['elong'] >= min_elong]
    return streaks, info


def _sample(img, xs, ys):
    return cv2.remap(img, xs.astype(numpy.float32), ys.astype(numpy.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def profile(d, p1, p2, half_width=3):
    """Brightness along the streak: max across +-half_width px at each step."""
    n = max(int(numpy.hypot(*(p2 - p1))), 2)
    t = numpy.linspace(0.0, 1.0, n)
    dirv = (p2 - p1) / numpy.hypot(*(p2 - p1))
    nrm = numpy.array([-dirv[1], dirv[0]])
    offs = numpy.arange(-half_width, half_width + 1)
    xs = p1[0] + numpy.outer(offs, numpy.ones(n)) * nrm[0] + numpy.outer(numpy.ones(len(offs)), t) * (p2[0] - p1[0])
    ys = p1[1] + numpy.outer(offs, numpy.ones(n)) * nrm[1] + numpy.outer(numpy.ones(len(offs)), t) * (p2[1] - p1[1])
    vals = _sample(d, xs, ys)
    return vals.max(axis=0), vals


def _runs(mask1d):
    m = numpy.concatenate([[0], mask1d.astype(numpy.int8), [0]])
    return int((numpy.diff(m) == 1).sum())


def measure(d, img_cur, img_ref, streak, sigma):
    """Shape and brightness features of one streak."""
    p1, p2 = streak['p1'], streak['p2']
    prof, across = profile(d, p1, p2)
    n = len(prof)
    snr = prof / sigma
    smooth = numpy.convolve(snr, numpy.ones(5) / 5, mode='same') if n >= 7 else snr
    peak = float(smooth.max())
    above = snr > 2.0
    k = max(int(0.15 * n), 2)
    head, tail = float(smooth[:k].mean()), float(smooth[-k:].mean())
    mid = smooth[k:-k] if n > 2 * k + 2 else smooth
    if not len(mid):
        mid = smooth
    # dashes: periodic on/off along the trail (aircraft strobes)
    dashed = 0.0
    if n >= 24:
        x = above.astype(numpy.float64) - above.mean()
        if x.std() > 0:
            ac = numpy.correlate(x, x, mode='full')[n - 1:]
            ac = ac / ac[0]
            lags = ac[4:max(n // 3, 5)]
            if len(lags):
                dashed = float(lags.max())
    # beads: aircraft strobes and nav lights give many evenly spaced peaks
    # along the trail; a meteor rises and fades smoothly (a flare or two)
    from scipy.signal import find_peaks
    raw = snr - numpy.convolve(snr, numpy.ones(15) / 15, mode='same') if n >= 20 else snr * 0
    # prominence relative to the trail's brightness: strobe beads stand well
    # clear of the gaps between them; ripples on a bright smooth trail don't
    peaks, _ = find_peaks(snr, prominence=max(3.0, 0.3 * float(snr.max())), distance=3)
    beads = len(peaks)
    bead_ac = 0.0
    if n >= 30 and raw.std() > 0:
        ac = numpy.correlate(raw, raw, mode='full')[n - 1:]
        ac = ac / ac[0]
        bead_ac = float(ac[3:21].max())
    # ripple: how far the profile swings around its own smoothed shape, in
    # noise units and relative to its level. Strobe beads swing from bright
    # to near-dark every few px; a meteor's profile is smooth.
    if n >= 15:
        sm9 = numpy.convolve(snr, numpy.ones(9) / 9, mode='same')[4:-4]
        resid = (snr[4:-4] - sm9)
        lit = sm9 > 2.0
        ripple = float(resid[lit].std()) if lit.sum() >= 5 else 0.0
        ripple_rel = ripple / float(sm9[lit].mean()) if lit.sum() >= 5 else 0.0
    else:
        ripple = ripple_rel = 0.0
    # sides: new light in parallel bands 6-10 px either side, relative to the
    # line. A moonlit cloud edge has diffuse glow beside it; a meteor or
    # aircraft is a clean line on dark sky.
    dirv = (p2 - p1) / max(float(numpy.hypot(*(p2 - p1))), 1e-6)
    nrm = numpy.array([-dirv[1], dirv[0]])
    side_vals = []
    for off in (-8, 8):
        sp1, sp2 = p1 + nrm * off, p2 + nrm * off
        sprof, _ = profile(d, sp1, sp2, half_width=2)
        side_vals.append(float(numpy.clip(sprof, 0, None).mean()))
    line_level = float(numpy.clip(prof, 0, None).mean()) or 1e-6
    side_ratio = max(side_vals) / line_level
    # local activity: share of the sky in a box around the streak (the streak
    # itself excluded) that changed by more than 3 sigma either way between
    # frames. Moonlit cloud drifting past changes everywhere around its
    # bright edges; a meteor on clear sky has next to nothing changing nearby.
    pad = 25
    h_img, w_img = d.shape
    bx1, bx2 = int(max(min(p1[0], p2[0]) - pad, 0)), int(min(max(p1[0], p2[0]) + pad, w_img))
    by1, by2 = int(max(min(p1[1], p2[1]) - pad, 0)), int(min(max(p1[1], p2[1]) + pad, h_img))
    box = d[by1:by2, bx1:bx2]
    yy, xx = numpy.mgrid[by1:by2, bx1:bx2]
    seg = p2 - p1
    tt = numpy.clip(((xx - p1[0]) * seg[0] + (yy - p1[1]) * seg[1]) / max(float(seg @ seg), 1e-6), 0, 1)
    dist = numpy.hypot(xx - (p1[0] + tt * seg[0]), yy - (p1[1] + tt * seg[1]))
    changed = numpy.abs(box) > 3.0 * sigma
    # a bright meteor's own glow spreads well past 5 px and past its fitted
    # ends: drop any brightening connected to the line itself
    _, lab = cv2.connectedComponents(((box > 3.0 * sigma) | (dist <= 5)).astype(numpy.uint8), connectivity=8)
    own = numpy.unique(lab[dist <= 2])
    glow = numpy.isin(lab, own[own > 0])
    around = (dist > 5) & (box != 0) & ~glow
    local_activity = float(changed[around].mean()) if around.any() else 0.0
    # integrated brightness of the streak pixels (in sigma units)
    flux = float(numpy.clip(d[streak['ys'], streak['xs']], 0, None).sum() / sigma)
    # colour of the new light, sampled along the line
    t = numpy.linspace(0, 1, n)
    lx = numpy.clip((p1[0] + t * (p2[0] - p1[0])).round().astype(int), 0, d.shape[1] - 1)
    ly = numpy.clip((p1[1] + t * (p2[1] - p1[1])).round().astype(int), 0, d.shape[0] - 1)
    col = numpy.clip(img_cur[ly, lx].astype(numpy.float32) - img_ref[ly, lx].astype(numpy.float32), 0, None)
    col = col[above] if above.any() else col
    bgr = col.mean(axis=0) if len(col) else numpy.zeros(3)
    tot = float(bgr.sum()) or 1.0
    # colour changes along the trail: red/green navigation lights
    if len(col) >= 5:
        s = col.sum(axis=1) + 1e-3
        bright = s > numpy.percentile(s, 50)
        rg = (col[:, 2] - col[:, 1]) / s
        colour_var = float(rg[bright].std()) if bright.sum() >= 3 else 0.0
    else:
        colour_var = 0.0
    # width: FWHM of the across-track profile at the brightest point
    ip = int(numpy.argmax(smooth))
    cross = across[:, max(ip - 2, 0):ip + 3].mean(axis=1)
    fwhm = float((cross > cross.max() / 2).sum()) if cross.max() > 0 else 0.0
    return {
        'length_px': round(streak['len'], 1),
        'width_px': round(streak['width'], 1),
        'fwhm_px': fwhm,
        'elong': round(streak['elong'], 1),
        'nfrag': streak['nfrag'],
        'peak_snr': round(peak, 1),
        'mean_snr': round(float(mid.mean()), 1),
        'coverage': round(float(above.mean()), 2),
        'runs': _runs(above),
        'dashed': round(dashed, 2),
        'taper': round(1.0 - (head + tail) / 2.0 / peak, 2) if peak > 0 else 0.0,
        'taper_min': round(1.0 - min(head, tail) / peak, 2) if peak > 0 else 0.0,
        'flat_cv': round(float(mid.std() / mid.mean()), 2) if mid.mean() > 0 else 0.0,
        'flux': round(flux, 0),
        'colour_bgr': [round(float(v) / tot, 2) for v in bgr],
        'colour_var': round(colour_var, 3),
        'beads': beads,
        'bead_density': round(beads / max(n, 1) * 100.0, 1),   # peaks per 100 px
        'bead_ac': round(bead_ac, 2),
        'ripple': round(ripple, 2),
        'ripple_rel': round(ripple_rel, 2),
        'side_ratio': round(side_ratio, 2),
        'local_activity': round(local_activity, 3),
    }


def extension_signal(d, sigma, points, kernel=7, nearest=30, control_px=14):
    """How strongly a neighbour frame's new light `d` lines up with the path
    beyond a streak's end: the fraction of the `nearest` of `points` (N x 2
    pixel coords, ordered outward from the end) above 3 sigma within
    +-kernel/2 px, minus the same fraction on parallel control paths
    control_px either side. A satellite or aircraft track carrying straight
    on lights the path but not its sides; drifting moonlit cloud lights both.
    Beaded trails count because the max filter bridges the gaps between
    beads; only the stretch next to the end is used. -> (excess, n)."""
    points = points[:nearest]
    if len(points) < 10:
        return 0.0, len(points)
    dm = cv2.dilate(d, numpy.ones((kernel, kernel), numpy.uint8))

    def lit(p):
        vals = _sample(dm, p[:, 0][None, :].astype(numpy.float32), p[:, 1][None, :].astype(numpy.float32))[0]
        return float((vals > 3.0 * sigma).mean())

    step = numpy.gradient(points, axis=0)
    step /= numpy.linalg.norm(step, axis=1, keepdims=True) + 1e-9
    side = numpy.stack([-step[:, 1], step[:, 0]], axis=1) * control_px
    control = max(lit(points + side), lit(points - side))
    return lit(points) - control, len(points)
