"""Meteor scorer: CLI and pipeline. Runs on the Pi with indi-allsky's Python
(numpy, opencv, sgp4, ephem are already in its virtualenv):

  PY=~/indi-allsky/virtualenv/indi-allsky/bin/python
  $PY -m allsky_meteors.scorer run              new night frames since last run
  $PY -m allsky_meteors.scorer night 2026-11-21 [--force]   one night (dayDate)
  $PY -m allsky_meteors.scorer frame IMAGE_ID   one frame, verbose
  $PY -m allsky_meteors.scorer inject 2026-11-21 [--n 300] [--synthetic]  recall test
  $PY -m allsky_meteors.scorer report [2026-11-21]

For each night frame (with its neighbours): find streaks of new light
(detect.py); reject ones that continue into a neighbouring frame (satellites,
aircraft), lie on a known satellite's path (satellites.py) or look like
aircraft strobes; measure the rest and give each

  conf       0-100, how meteor-like (class meteor >= 65, possible >= 45)
  spectacle  0-100, how good a meteor: brightness, length, altitude, shower

Results go to ~/allsky-meteors/meteors.sqlite with a crop and a marked full
frame per streak under ~/allsky-meteors/YYYY-MM-DD/.
"""

import argparse
import json
import logging
import math
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy

from . import astro, db, detect, showers
from .lens import angle_between, from_indi_config

logger = logging.getLogger('allsky_meteors')

PROFILES = Path(__file__).resolve().parents[1] / 'data' / 'nsucp_profiles.npy'
HOME = Path(os.environ.get('ALLSKY_METEORS_HOME', Path.home() / 'allsky-meteors'))
MAX_NEIGHBOUR_GAP_S = 20.0   # beyond the neighbour's own exposure
CONT_FRACTION = 0.4          # of the 6 deg beyond an end lit in a neighbour = moving object
BEADS_MIN = 5                # aircraft: this many strobe peaks...
BEAD_DENSITY_MIN = 3.0       # ...at least this many per 100 px
COLOUR_VAR_MIN = 0.15        # aircraft: red/green alternation along the trail
RIPPLE_REL_MIN = 0.25        # aircraft: strobe beads (profile swing / level)...
RIPPLE_MIN = 1.5             # ...and in noise units, so faint noisy trails don't count
SIDE_RATIO_MAX = 0.5         # cloud: glow beside the line this bright relative to it
LOCAL_ACTIVITY_MAX = 0.1     # cloud: share of the sky around the streak that changed
                             # (2026-11-21 moonlit wisps 0.16-0.91; meteor 0.005)
TWILIGHT_SUN_ALT = -12.0     # sun higher than this: sky too bright to show off a meteor
EDGE_PX = 20                 # streak this close to the mask edge (trees, house, rim)
TRACK_WINDOW_S = 40          # streaks in other frames this close in time...
TRACK_ANGLE_DEG = 15         # ...at about the same direction (fisheye curvature)...
TRACK_PERP_PX = 10           # ...and on the same line...
TRACK_NEAR_PX = 300          # ...and not far along it are one moving object
EVEN_TAPER = 0.5             # meteor class needs this much fade...
EVEN_FLAT_CV = 0.25          # ...or this much brightness variation
METEOR_CONF = 65
POSSIBLE_CONF = 45


# --------------------------------------------------------------------------- setup

def indi_config():
    """indi-allsky's settings, or on a SyncAPI receiver (whose own config
    isn't the camera's) the JSON file ALLSKY_METEORS_CONFIG with the keys
    used here: LOCATION_LATITUDE/LONGITUDE/ELEVATION, IMAGE_FOLDER,
    DETECT_MASK, IMAGE_FLIP_V/H, LENS_AZIMUTH, VIRTUALSKY offsets."""
    path = os.environ.get('ALLSKY_METEORS_CONFIG')
    if path:
        return json.loads(Path(path).read_text())
    return json.loads(db.query('select data from config order by id desc limit 1')[0]['data'])


def open_db():
    HOME.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(HOME / 'meteors.sqlite')
    con.executescript('''
        create table if not exists processed (
            image_id integer primary key, night text, frame_time text, status text,
            n_streaks integer, ms integer);
        create table if not exists detections (
            id integer primary key, image_id integer, night text, frame_time text, file text,
            x1 real, y1 real, x2 real, y2 real, cls text, reason text, conf integer, spectacle integer,
            shower text, features text, crop text, marked text, published integer default 0,
            label text);
        create index if not exists det_night on detections(night);
    ''')
    return con


class Context:
    def __init__(self):
        self.cfg = indi_config()
        self.lat = float(self.cfg['LOCATION_LATITUDE'])
        self.lon = float(self.cfg['LOCATION_LONGITUDE'])
        self.elev = float(self.cfg.get('LOCATION_ELEVATION', 0))
        self.image_dir = Path(self.cfg['IMAGE_FOLDER'])
        self._mask = None
        self.lens = None
        self._sats = None
        self._cache = {}

    def mask(self, shape):
        if self._mask is None:
            path = self.cfg.get('DETECT_MASK')
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
            if m is None:
                raise SystemExit(f'no detection mask at {path!r}: the scorer needs DETECT_MASK')
            # the mask is in raw sensor orientation; saved frames are flipped
            # both ways (IMAGE_FLIP_V + IMAGE_FLIP_H = 180 degrees)
            if self.cfg.get('IMAGE_FLIP_V'):
                m = cv2.flip(m, 0)
            if self.cfg.get('IMAGE_FLIP_H'):
                m = cv2.flip(m, 1)
            m = cv2.erode(m, numpy.ones((7, 7), numpy.uint8))
            self._mask = m > 127
            self.mask_idx = numpy.flatnonzero(self._mask)[::7]
            self.edge_dist = cv2.distanceTransform(self._mask.astype(numpy.uint8), cv2.DIST_L2, 5)
            self.lens = from_indi_config(self.cfg, shape[1], shape[0])
        return self._mask

    @property
    def sats(self):
        if self._sats is None:
            from .satellites import SatelliteVeto
            self._sats = SatelliteVeto(HOME / 'tle-active.txt', self.lat, self.lon, self.elev)
        return self._sats

    def load(self, row):
        """Decoded frame (BGR, gray) for an image row, cached for neighbours."""
        if row['id'] in self._cache:
            return self._cache[row['id']]
        img = cv2.imread(str(self.image_dir / row['filename']))
        val = None if img is None else (img, detect.gray(img))
        self._cache[row['id']] = val
        if len(self._cache) > 4:
            self._cache.pop(min(self._cache))
        return val


def frame_rows(where, args=()):
    return db.query('select id, filename, createDate, exposure, gain, moonmode, detections, dayDate, camera_id '
                    f'from image where night = 1 and {where} order by id', args)


def frame_end(row):
    """createDate is local time when the frame was read out (end of exposure):
    the Pi's, or a SyncAPI receiver's (it shifts times into its own zone), so
    run with TZ set to match."""
    return datetime.fromisoformat(row['createDate']).astimezone()


def adjacent(a, b):
    """Are rows a and b consecutive exposures (b after a)?"""
    gap = (frame_end(b) - frame_end(a)).total_seconds()
    return 0 < gap <= float(b['exposure']) + MAX_NEIGHBOUR_GAP_S


# --------------------------------------------------------------------------- analysis

def _vecs(ctx, s):
    return ctx.lens.to_vec(*s['p1']), ctx.lens.to_vec(*s['p2'])


def _along(v_start, normal, w):
    return math.degrees(math.atan2(float(numpy.dot(normal, numpy.cross(v_start, w))), float(numpy.dot(v_start, w))))


def extension_points(ctx, s, mask, ext_deg, gap_deg=0.4, step_deg=0.2):
    """Pixels along the streak's great circle beyond each end, out to
    ext_deg, inside the mask: {'start': Nx2, 'end': Nx2}."""
    v1, v2 = _vecs(ctx, s)
    normal = numpy.cross(v1, v2)
    if numpy.linalg.norm(normal) < 1e-9:
        return {}
    normal /= numpy.linalg.norm(normal)
    toward = numpy.cross(normal, v1)       # from v1 towards v2
    length = float(angle_between(v1, v2))
    ext, gap, step = (math.radians(v) for v in (ext_deg, gap_deg, step_deg))
    h, w = mask.shape
    out = {}
    for name, thetas in (('end', numpy.arange(length + gap, length + ext, step)),
                         ('start', -numpy.arange(gap, ext, step))):
        vec = numpy.outer(numpy.cos(thetas), v1) + numpy.outer(numpy.sin(thetas), toward)
        from .lens import vec_to_altaz
        alt, az = vec_to_altaz(vec)
        x, y = ctx.lens.to_xy(alt, az)
        xi, yi = numpy.round(x).astype(int), numpy.round(y).astype(int)
        ok = (alt > 0) & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok[ok] = mask[yi[ok], xi[ok]]
        out[name] = numpy.stack([x[ok], y[ok]], axis=1)
    return out


def continuation(ctx, s, mask, nb_diffs):
    """Does new light in a neighbouring frame carry on along this streak's
    great circle beyond either end? A satellite or aircraft moves on into the
    next frame (and came from the previous one); a meteor doesn't.
    -> description or None."""
    length = math.degrees(float(angle_between(*_vecs(ctx, s))))
    pts = extension_points(ctx, s, mask, ext_deg=max(length, 6.0))
    best = {}
    for label, dn, sn in nb_diffs:
        for end, p in pts.items():
            frac, n = detect.extension_signal(dn, sn, p)
            best[f'{label[0]}{end[0]}'] = (round(frac, 2), n)
            if n >= 10 and frac >= CONT_FRACTION:
                return f'continues past its {end} in the {label} frame ({frac:.0%} of {n} px)', best
    return None, best


def parallel_pairs(streaks):
    """Indices of streaks with a close parallel companion: an aircraft's
    navigation lights draw two or three side by side."""
    flagged = set()
    for i, a in enumerate(streaks):
        da = (a['p2'] - a['p1']) / a['len']
        for j in range(i + 1, len(streaks)):
            b = streaks[j]
            db = (b['p2'] - b['p1']) / b['len']
            if abs(float(numpy.dot(da, db))) < math.cos(math.radians(4)):
                continue
            mid_b = (b['p1'] + b['p2']) / 2
            off = mid_b - a['p1']
            perp = abs(float(off[0] * -da[1] + off[1] * da[0]))
            if 2.0 < perp < 30.0:
                flagged.update((i, j))
    return flagged


def moon_altaz(ctx, when):
    import ephem
    o = ephem.Observer()
    o.lat, o.lon, o.elevation = str(ctx.lat), str(ctx.lon), ctx.elev
    o.date = when.astimezone(timezone.utc).replace(tzinfo=None)
    m = ephem.Moon(o)
    return float(m.alt), float(m.az), float(m.phase)


def clip01(x):
    return min(max(x, 0.0), 1.0)


def score(f):
    """Heuristic meteor confidence and spectacle from features. Weights are a
    starting point, tuned on the 2026-11-20 night (one meteor among ~40
    aircraft and cloud lines), to be replaced by a classifier trained on
    labels.

    Taper carries most weight: a meteor brightens and fades within the
    exposure, while an aircraft's steady light (or a satellite's) runs at an
    even brightness to where the exposure cut it off."""
    c = 40.0
    c += 30 * clip01((f['taper'] - 0.3) / 0.4)           # faded ends
    c += 10 * clip01((f['elong'] - 5) / 20)              # thin and straight
    c += 10 * clip01((f['peak_snr'] - 5) / 20)
    c += 10 if f.get('shower') else 0
    c -= 20 * clip01((f['ripple_rel'] - 0.15) / 0.1)     # strobe ripple
    c -= 20 * clip01((f['dashed'] - 0.2) / 0.3)          # dashes
    c -= 15 * clip01((f['side_ratio'] - 0.25) / 0.25)    # cloud glow alongside
    c -= 10 if f['coverage'] < 0.6 else 0                # broken up
    c -= 10 if f['fwhm_px'] > 8 else 0                   # fat: cloud edge, glow
    c -= 10 if f.get('moon_sep_deg', 180) < 20 else 0    # Moon glare
    c -= 15 if f.get('edge_px', 99) < EDGE_PX else 0      # against the trees / mask edge
    conf = int(round(min(max(c, 0), 100)))
    sp = (35 * clip01(math.log10(max(f['flux'], 1)) / 4) + 30 * clip01(f['length_deg'] / 40)
          + 15 * clip01(f['peak_snr'] / 60) + 10 * clip01(f['alt_mid_deg'] / 60) + (10 if f.get('shower') else 0))
    return conf, int(round(sp))


def shape_veto(f):
    """Vetoes that need only the streak's own features (no neighbouring
    frames): aircraft strobes and lights, cloud. -> (cls, reason) or
    (None, None)."""
    if f['beads'] >= BEADS_MIN and f['bead_density'] >= BEAD_DENSITY_MIN:
        return 'aircraft', f'beaded trail ({f["beads"]} strobe peaks)'
    if f['colour_var'] >= COLOUR_VAR_MIN and f['beads'] >= 3:
        return 'aircraft', f'red/green lights (colour variation {f["colour_var"]})'
    if f['ripple_rel'] >= RIPPLE_REL_MIN and f['ripple'] >= RIPPLE_MIN:
        return 'aircraft', f'strobe ripple ({f["ripple_rel"]})'
    if f['local_activity'] >= LOCAL_ACTIVITY_MAX:
        return 'cloud', f'sky changing around it ({f["local_activity"]:.0%})'
    if f['side_ratio'] >= SIDE_RATIO_MAX:
        return 'cloud', f'glow beside the line ({f["side_ratio"]})'
    if f['dashed'] >= 0.4 and f['runs'] >= 4:
        return 'aircraft', f'dashed trail (periodicity {f["dashed"]})'
    return None, None


def edge_px(ctx, s, n=9):
    """Closest approach of the streak to the mask edge, px."""
    t = numpy.linspace(0, 1, n)
    xs = numpy.clip((s['p1'][0] + t * (s['p2'][0] - s['p1'][0])).round().astype(int), 0, ctx.edge_dist.shape[1] - 1)
    ys = numpy.clip((s['p1'][1] + t * (s['p2'][1] - s['p1'][1])).round().astype(int), 0, ctx.edge_dist.shape[0] - 1)
    return round(float(ctx.edge_dist[ys, xs].min()), 1)


def link_tracks(db, night):
    """A meteor or possible that lines up with a streak in another frame
    within TRACK_WINDOW_S is one object moving across frames (a satellite or
    aircraft whose exposures leave gaps too long for the continuation test,
    e.g. 6 s exposures every 15 s at dusk). Reclasses them as moving."""
    rows = db.execute('select id, image_id, frame_time, x1, y1, x2, y2, cls from detections where night = ?'
                      ' order by frame_time', (night,)).fetchall()
    t = [datetime.fromisoformat(r[2]).timestamp() for r in rows]
    cos_tol = math.cos(math.radians(TRACK_ANGLE_DEG))
    n = 0
    for i, r in enumerate(rows):
        if r[7] not in ('meteor', 'possible'):
            continue
        a, b = numpy.array(r[3:5]), numpy.array(r[5:7])
        u = (b - a) / max(float(numpy.linalg.norm(b - a)), 1e-6)
        for j, o in enumerate(rows):
            if o[1] == r[1] or abs(t[j] - t[i]) > TRACK_WINDOW_S:
                continue
            oa, ob = numpy.array(o[3:5]), numpy.array(o[5:7])
            ou = (ob - oa) / max(float(numpy.linalg.norm(ob - oa)), 1e-6)
            if abs(float(u @ ou)) < cos_tol:
                continue
            perp = max(abs(float((p - a) @ numpy.array([-u[1], u[0]]))) for p in (oa, ob))
            near = min(float(numpy.linalg.norm(p - q)) for p in (oa, ob) for q in (a, b))
            if perp <= TRACK_PERP_PX and near <= TRACK_NEAR_PX:
                db.execute("update detections set cls = 'moving', conf = 0, reason = ? where id = ?",
                           (f'on one track with #{o[0]} ({o[2][11:19]})', r[0]))
                n += 1
                break
    db.commit()
    return n


def analyse(ctx, row, prev_row, next_row, use_sats=True, frame=None):
    """Streaks in one frame -> (status, [detection dicts]). `frame` overrides
    the decoded current frame (injection tests)."""
    cur = frame or ctx.load(row)
    if cur is None:
        return 'missing file', []
    img, g = cur
    mask = ctx.mask(g.shape)
    nbrs = []
    for label, r in (('previous', prev_row), ('next', next_row)):
        if r is not None:
            v = ctx.load(r)
            if v is not None:
                nbrs.append((label, r, v))
    if not nbrs:
        return 'no neighbours', []
    d, scales = detect.new_light(g, [v[1] for _, _, v in nbrs], ctx.mask_idx)
    if d is None:
        return scales, []
    sigma = detect.robust_sigma(d, ctx.mask_idx)
    streaks, info = detect.detect(d, mask, sigma)
    if streaks is None:
        return f'skipped: {info}', []
    if not streaks:
        return 'ok', []

    # what's new in each neighbour relative to this frame: the continuation
    # of a satellite or aircraft track shows up there
    nb_diffs = []
    for (label, r, v), s in zip(nbrs, scales):
        dn = v[1] * s - g
        nb_diffs.append((label, dn, detect.robust_sigma(dn, ctx.mask_idx)))
    ref_img = numpy.maximum.reduce([v[0] for _, _, v in nbrs])
    plane_pairs = parallel_pairs(streaks)
    when = frame_end(row)
    mid_time = when - timedelta(seconds=float(row['exposure']) / 2)
    sra, sdec = astro.sun_radec(mid_time)
    sun_alt = round(math.degrees(float(astro.radec_to_altaz(sra, sdec, mid_time, ctx.lat, ctx.lon)[0])), 1)
    moon_alt, moon_az, moon_phase = moon_altaz(ctx, mid_time)
    from .lens import altaz_to_vec
    moon_vec = altaz_to_vec(moon_alt, moon_az)
    active = showers.active(mid_time.date())

    out = []
    for i, s in enumerate(sorted(streaks, key=lambda s: -s['len'])[:20]):
        f = detect.measure(d, img, ref_img, s, sigma)
        v1, v2 = _vecs(ctx, s)
        alt_mid, az_mid = ctx.lens.to_altaz(*((s['p1'] + s['p2']) / 2))
        f['length_deg'] = round(math.degrees(float(angle_between(v1, v2))), 1)
        f['alt_mid_deg'] = round(math.degrees(float(alt_mid)), 1)
        f['az_mid_deg'] = round(math.degrees(float(az_mid)), 1)
        mid_vec = ctx.lens.to_vec(*((s['p1'] + s['p2']) / 2))
        f['moon_sep_deg'] = round(math.degrees(float(angle_between(mid_vec, moon_vec))), 1) if moon_alt > 0 else 180.0
        f['moonmode'] = bool(row['moonmode'])
        f['indi_detections'] = row['detections']
        f['sigma'] = round(sigma, 2)
        f['sun_alt_deg'] = sun_alt
        f['edge_px'] = edge_px(ctx, s)
        sh = showers.associate(v1, v2, mid_time, ctx.lat, ctx.lon, active)
        f['shower'] = sh['code'] if sh else None
        f['shower_off_deg'] = sh['off_deg'] if sh else None

        reason, f['cont'] = continuation(ctx, s, mask, nb_diffs)
        cls = 'moving' if reason else None
        if not cls and i in plane_pairs:
            cls, reason = 'aircraft', 'parallel companion streak (navigation lights)'
        if not cls:
            cls, reason = shape_veto(f)
        if not cls and use_sats:
            m = ctx.sats.match(v1, v2, when, float(row['exposure']))
            if m:
                cls, reason = 'satellite', f'{m[0]} (off {m[1]} deg)'
        conf, spect = score(f)
        if cls:
            conf = 0
        else:
            cls = 'meteor' if conf >= METEOR_CONF else 'possible' if conf >= POSSIBLE_CONF else 'junk'
            reason = ''
            if cls == 'meteor' and sun_alt > TWILIGHT_SUN_ALT:
                cls, reason = 'possible', f'twilight (sun {sun_alt} deg)'
            # a showable meteor fades or flares along its trail; an even
            # streak is more likely a steady light (tentative: from labels
            # 2026-11-22, one even 15 deg streak vs two good meteors)
            if cls == 'meteor' and f['taper'] < EVEN_TAPER and f['flat_cv'] < EVEN_FLAT_CV:
                cls, reason = 'possible', f'even brightness (taper {f["taper"]}, cv {f["flat_cv"]})'
        out.append({'p1': s['p1'], 'p2': s['p2'], 'cls': cls, 'reason': reason, 'conf': conf,
                    'spectacle': spect, 'features': f})
    return 'ok', out


# --------------------------------------------------------------------------- output

def stretch(img):
    lo, hi = numpy.percentile(img, (1.0, 99.8))
    return numpy.clip((img.astype(numpy.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(numpy.uint8)


def save_images(ctx, row, det, night_dir, n):
    img = ctx.load(row)[0]
    h, w = img.shape[:2]
    (x1, y1), (x2, y2) = det['p1'], det['p2']
    pad = 40
    bx1, bx2 = int(max(min(x1, x2) - pad, 0)), int(min(max(x1, x2) + pad, w))
    by1, by2 = int(max(min(y1, y2) - pad, 0)), int(min(max(y1, y2) + pad, h))
    stem = f'{Path(row["filename"]).stem}_{n}'
    crop = night_dir / f'{stem}_crop.jpg'
    cv2.imwrite(str(crop), stretch(img[by1:by2, bx1:bx2]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    marked = img.copy()
    colour = {'meteor': (0, 255, 255), 'possible': (0, 200, 255)}.get(det['cls'], (128, 128, 128))
    m = 12
    for (cx, cy, sx, sy) in ((bx1, by1, 1, 1), (bx2, by1, -1, 1), (bx1, by2, 1, -1), (bx2, by2, -1, -1)):
        cv2.line(marked, (cx, cy), (cx + sx * m, cy), colour, 2)
        cv2.line(marked, (cx, cy), (cx, cy + sy * m), colour, 2)
    mk = night_dir / f'{stem}_marked.jpg'
    cv2.imwrite(str(mk), cv2.resize(marked, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(crop), str(mk)


def record(ctx, db, row, status, dets, ms):
    night = row['dayDate']
    db.execute('delete from detections where image_id = ?', (row['id'],))
    night_dir = HOME / night
    for n, det in enumerate(dets):
        crop = marked = None
        if det['cls'] in ('meteor', 'possible') or det['features']['length_deg'] >= 8:
            night_dir.mkdir(parents=True, exist_ok=True)
            crop, marked = save_images(ctx, row, det, night_dir, n)
        f = det['features']
        db.execute('insert into detections (image_id, night, frame_time, file, x1, y1, x2, y2, cls, reason, conf,'
                   ' spectacle, shower, features, crop, marked) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (row['id'], night, frame_end(row).isoformat(), row['filename'],
                    float(det['p1'][0]), float(det['p1'][1]), float(det['p2'][0]), float(det['p2'][1]),
                    det['cls'], det['reason'], det['conf'], det['spectacle'], f.get('shower'),
                    json.dumps(f), crop, marked))
    db.execute('insert or replace into processed values (?,?,?,?,?,?)',
               (row['id'], night, frame_end(row).isoformat(), status, len(dets), ms))


def process(ctx, db, rows, verbose=False):
    """Process every row; neighbours come from the same list."""
    counts = {}
    for i, row in enumerate(rows):
        prev_row = rows[i - 1] if i > 0 and adjacent(rows[i - 1], row) else None
        next_row = rows[i + 1] if i + 1 < len(rows) and adjacent(row, rows[i + 1]) else None
        t0 = time.monotonic()
        status, dets = analyse(ctx, row, prev_row, next_row)
        ms = int((time.monotonic() - t0) * 1000)
        record(ctx, db, row, status, dets, ms)
        for d in dets:
            counts[d['cls']] = counts.get(d['cls'], 0) + 1
            if verbose or d['cls'] in ('meteor', 'possible'):
                logger.info('%s %s conf %d spectacle %d %s %s', row['createDate'][:19], d['cls'], d['conf'],
                            d['spectacle'], d['reason'], json.dumps(d['features']))
        if i % 200 == 0:
            db.commit()
    db.commit()
    return counts


# --------------------------------------------------------------------------- commands

def cmd_run(args):
    """New frames since the last run. The newest frame waits for its next
    neighbour, so a meteor in it is found on the following run."""
    ctx = Context()
    db = open_db()
    last = db.execute('select max(image_id) from processed').fetchone()[0] or 0
    if last:
        # the last processed frame comes along as the first one's previous
        # neighbour
        rows = frame_rows('id >= ?', (last,))
        start = 1 if rows and rows[0]['id'] == last else 0
    else:
        rows = frame_rows('1')[-args.limit:]  # first run: don't trawl the archive
        start = 0
    # the newest frame is held back until its successor exists
    if len(rows) - 1 <= start:
        logger.info('run: no new frames')
        return
    counts = _process_window(ctx, db, rows, start, args.verbose)
    for night in sorted({r['dayDate'] for r in rows[start:-1]}):
        link_tracks(db, night)
    logger.info('run: %d frames, %s', len(rows) - 1 - start, counts)


def _process_window(ctx, db, rows, start, verbose):
    """Process rows[start:-1], using rows[start-1] and rows[-1] as context."""
    counts = {}
    for i in range(start, len(rows) - 1):
        row = rows[i]
        prev_row = rows[i - 1] if i > 0 and adjacent(rows[i - 1], row) else None
        next_row = rows[i + 1] if adjacent(row, rows[i + 1]) else None
        t0 = time.monotonic()
        status, dets = analyse(ctx, row, prev_row, next_row)
        record(ctx, db, row, status, dets, int((time.monotonic() - t0) * 1000))
        for d in dets:
            counts[d['cls']] = counts.get(d['cls'], 0) + 1
            if verbose or d['cls'] in ('meteor', 'possible'):
                logger.info('%s %s conf %d spectacle %d %s %s', row['createDate'][:19], d['cls'], d['conf'],
                            d['spectacle'], d['reason'], json.dumps(d['features']))
        if i % 200 == 0:
            db.commit()
    db.commit()
    return counts


def cmd_night(args):
    ctx = Context()
    db = open_db()
    rows = frame_rows('dayDate = ?', (args.night,))
    if not rows:
        raise SystemExit(f'no night frames for dayDate {args.night}')
    if not args.force:
        done = {r[0] for r in db.execute('select image_id from processed where night = ?', (args.night,))}
        if done and len(done) >= len(rows) - 1:
            raise SystemExit(f'{args.night} already processed ({len(done)} frames); use --force')
    t0 = time.monotonic()
    counts = process(ctx, db, rows, verbose=args.verbose)
    logger.info('night %s: %d frames in %.0f s: %s', args.night, len(rows), time.monotonic() - t0, counts)
    logger.info('night %s: %d on a track across frames', args.night, link_tracks(db, args.night))
    cmd_report(argparse.Namespace(night=args.night, top=args.top))


def cmd_frame(args):
    ctx = Context()
    rows = frame_rows('id between ? and ?', (args.image_id - 1, args.image_id + 1))
    by = {r['id']: r for r in rows}
    row = by.get(args.image_id)
    if row is None:
        raise SystemExit('no such night frame')
    prev_row = by.get(args.image_id - 1)
    next_row = by.get(args.image_id + 1)
    prev_row = prev_row if prev_row and adjacent(prev_row, row) else None
    next_row = next_row if next_row and adjacent(row, next_row) else None
    status, dets = analyse(ctx, row, prev_row, next_row)
    print(status)
    for d in dets:
        print(d['cls'], d['conf'], d['spectacle'], d['reason'], [round(float(v)) for v in (*d['p1'], *d['p2'])])
        print('   ', json.dumps(d['features']))


def cmd_report(args):
    db = open_db()
    where, params = ('where night = ?', (args.night,)) if args.night else ('', ())
    print('frames processed:', dict(db.execute(
        f'select status, count(*) from processed {where} group by status', params).fetchall()))
    print('detections by class:', dict(db.execute(
        f'select cls, count(*) from detections {where} group by cls', params).fetchall()))
    q = (f'select frame_time, cls, conf, spectacle, shower, reason, features, crop from detections '
         f'{where + " and" if where else "where"} cls in ("meteor","possible") order by conf desc, spectacle desc '
         f'limit ?')
    for t, cls, conf, sp, sh, reason, feats, crop in db.execute(q, (*params, args.top)):
        f = json.loads(feats)
        print(f'{t[:19]} {cls:8s} conf {conf:3d} spect {sp:3d} {sh or "-":4s} len {f["length_deg"]:5.1f}d '
              f'snr {f["peak_snr"]:5.1f} taper {f["taper"]:.2f} dashed {f["dashed"]:.2f} alt {f["alt_mid_deg"]:4.0f} {crop}')


def new_light_for(ctx, row):
    """The new-light difference image for a stored frame (as in analyse)."""
    rows = frame_rows('id between ? and ?', (row['id'] - 1, row['id'] + 1))
    by = {r['id']: r for r in rows}
    refs = []
    for nid, before in ((row['id'] - 1, True), (row['id'] + 1, False)):
        r = by.get(nid)
        if r and (adjacent(r, row) if before else adjacent(row, r)):
            v = ctx.load(r)
            if v is not None:
                refs.append(v[1])
    cur = ctx.load(row)
    if cur is None or not refs:
        return None
    ctx.mask(cur[1].shape)
    d, _ = detect.new_light(cur[1], refs, ctx.mask_idx)
    return d


def ml_crop(d, p1, p2, pad=10):
    """A crop in the form GMN's RMS MLFilter feeds its model: background-
    subtracted light around the detection, padded 10 px, made square with
    black fill, 8-bit greyscale."""
    h, w = d.shape
    x1, x2 = int(max(min(p1[0], p2[0]) - pad, 0)), int(min(max(p1[0], p2[0]) + pad + 1, w))
    y1, y2 = int(max(min(p1[1], p2[1]) - pad, 0)), int(min(max(p1[1], p2[1]) + pad + 1, h))
    c = numpy.clip(d[y1:y2, x1:x2], 0, 255).astype(numpy.uint8)
    side = max(c.shape)
    sq = numpy.zeros((side, side), numpy.uint8)
    oy, ox = (side - c.shape[0]) // 2, (side - c.shape[1]) // 2
    sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return sq


def cmd_mlcrops(args):
    """Write GMN-style crops (new light) for a night's detections into
    OUTDIR/<class>/<detection id>.png, for testing external classifiers."""
    ctx = Context()
    db = open_db()
    out = Path(args.outdir)
    q = 'select id, image_id, x1, y1, x2, y2, cls from detections where night = ?'
    params = [args.night]
    if args.cls:
        q += f' and cls in ({",".join("?" * len(args.cls))})'
        params += args.cls
    rows = {r['id']: r for r in frame_rows('dayDate = ?', (args.night,))}
    n = 0
    for did, image_id, x1, y1, x2, y2, cls in db.execute(q, params).fetchall():
        row = rows.get(image_id)
        if row is None:
            continue
        d = new_light_for(ctx, row)
        if d is None:
            continue
        (out / cls).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / cls / f'{did}.png'), ml_crop(d, numpy.array([x1, y1]), numpy.array([x2, y2])))
        n += 1
    print(f'{n} crops in {out}')


def cmd_inject(args):
    """Recall test: add synthetic meteors (single frame, faded ends) and
    satellites (continuing into the next frame) to real frames, and count
    what the pipeline finds and how it classes them. Each meteor takes a
    real brightness profile along its trail (rows of PROFILES, peak 1;
    tools/nsucp_eval.py extracts them from NightSkyUCP), or with --synthetic
    the old smooth rise and fade."""
    ctx = Context()
    profiles = None if args.synthetic else numpy.load(args.profiles)
    rows = frame_rows('dayDate = ?', (args.night,))
    rng = random.Random(args.seed)
    triples = [(rows[i - 1], rows[i], rows[i + 1]) for i in range(1, len(rows) - 1)
               if adjacent(rows[i - 1], rows[i]) and adjacent(rows[i], rows[i + 1])]
    sample = rng.sample(triples, min(args.n, len(triples)))
    amps = [4, 6, 10, 15, 25, 50]
    results = {}
    reasons = {}
    for k, (pr, row, nx) in enumerate(sample):
        base = ctx.load(row)
        if base is None or ctx.load(pr) is None or ctx.load(nx) is None:
            continue
        mask = ctx.mask(base[1].shape)
        kind = 'satellite' if k % 4 == 3 else 'meteor'
        amp = amps[k % len(amps)]
        ys, xs = numpy.nonzero(mask)
        j = rng.randrange(len(xs))
        x0, y0 = float(xs[j]), float(ys[j])
        ang = rng.uniform(0, math.pi)
        length = rng.uniform(25, 160)
        x1, y1 = x0 + math.cos(ang) * length, y0 + math.sin(ang) * length
        shape = None
        if kind == 'meteor':
            shape = _synthetic_shape() if profiles is None else profiles[rng.randrange(len(profiles))]
            if rng.random() < 0.5:
                shape = shape[::-1]
        img = _inject(base[0], (x0, y0), (x1, y1), amp, shape)
        frame = (img, detect.gray(img))
        if kind == 'satellite':
            # the next frame carries on along the same line
            nimg = _inject(ctx.load(nx)[0], (x1, y1), (x1 + (x1 - x0), y1 + (y1 - y0)), amp, None)
            ctx._cache[nx['id']] = (nimg, detect.gray(nimg))
        status, dets = analyse(ctx, row, pr, nx, use_sats=False, frame=frame)
        ctx._cache.pop(nx['id'], None)
        ctx._cache.pop(row['id'], None)
        hit = None
        for d in dets:
            if _matches(d, (x0, y0), (x1, y1)):
                hit = d
                break
        key = (kind, amp)
        r = results.setdefault(key, {'n': 0, 'found': 0, 'meteor': 0, 'possible': 0, 'vetoed': 0, 'skipped': 0})
        r['n'] += 1
        if status != 'ok':
            r['skipped'] += 1
        if hit:
            r['found'] += 1
            if hit['cls'] in ('meteor', 'possible'):
                r[hit['cls']] += 1
            elif hit['cls'] in ('moving', 'aircraft', 'satellite', 'cloud'):
                r['vetoed'] += 1
                if kind == 'meteor':
                    why = hit['reason'].split('(')[0].strip()
                    reasons[why] = reasons.get(why, 0) + 1
                    if args.verbose:
                        f = hit['features']
                        print(f'vetoed {row["id"]} {row["createDate"]} amp {amp} at ({x0:.0f},{y0:.0f}) len {length:.0f}: '
                              f'{hit["reason"]} activity {f["local_activity"]} side {f["side_ratio"]} sigma {f["sigma"]}')
    print('kind      amp  n  found  meteor possible vetoed skipped')
    for (kind, amp), r in sorted(results.items()):
        print(f'{kind:9s} {amp:3d} {r["n"]:3d} {r["found"]:5d} {r["meteor"]:7d} {r["possible"]:8d} '
              f'{r["vetoed"]:6d} {r["skipped"]:7d}')
    print('meteors wrongly vetoed:', reasons)


def _synthetic_shape(n=64):
    """Brightens then fades, peak 60 % along, ends near zero."""
    t = numpy.linspace(0, 1, n)
    phase = numpy.where(t < 0.6, t / 0.6 * 0.5, 0.5 + (t - 0.6) / 0.4 * 0.5)
    return numpy.sin(numpy.pi * phase) ** 1.5


def _inject(img, p1, p2, amp, shape):
    """Draw a soft line of peak `amp` (8-bit DN) from p1 to p2 with brightness
    `shape` along it (peak 1), or even brightness (a satellite) if None."""
    out = img.astype(numpy.float32)
    n = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])) * 3
    h, w = img.shape[:2]
    for i in range(n):
        t = i / max(n - 1, 1)
        b = 1.0 if shape is None else float(numpy.interp(t, numpy.linspace(0, 1, len(shape)), shape))
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        xi, yi = int(x), int(y)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx, yy = xi + dx, yi + dy
                if 0 <= xx < w and 0 <= yy < h:
                    wgt = math.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.8 ** 2)) / 3.0
                    out[yy, xx] += amp * b * wgt * numpy.array([0.9, 1.0, 0.85], numpy.float32)
    return numpy.clip(out, 0, 255).astype(numpy.uint8)


def _matches(det, p1, p2, tol=8.0):
    a = numpy.array(p1)
    b = numpy.array(p2)
    d = (b - a) / numpy.linalg.norm(b - a)
    for p in (det['p1'], det['p2']):
        off = p - a
        if abs(off[0] * -d[1] + off[1] * d[0]) > tol:
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog='allsky_meteors')
    ap.add_argument('-v', '--verbose', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--limit', type=int, default=400, help='frames on the very first run')
    p = sub.add_parser('night')
    p.add_argument('night')
    p.add_argument('--force', action='store_true')
    p.add_argument('--top', type=int, default=25)
    p = sub.add_parser('frame')
    p.add_argument('image_id', type=int)
    p = sub.add_parser('report')
    p.add_argument('night', nargs='?')
    p.add_argument('--top', type=int, default=25)
    p = sub.add_parser('mlcrops')
    p.add_argument('night')
    p.add_argument('outdir')
    p.add_argument('--cls', nargs='*')
    p = sub.add_parser('inject')
    p.add_argument('night')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--profiles', default=str(PROFILES), help='real meteor brightness profiles (.npy, rows peak 1)')
    p.add_argument('--synthetic', action='store_true', help='synthetic meteor shape instead of real profiles')
    p.add_argument('--verbose', action='store_true', help='list each wrongly vetoed meteor')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    {'run': cmd_run, 'night': cmd_night, 'frame': cmd_frame, 'report': cmd_report, 'inject': cmd_inject,
     'mlcrops': cmd_mlcrops}[args.cmd](args)


if __name__ == '__main__':
    main()
