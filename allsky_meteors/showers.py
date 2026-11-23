"""Major meteor showers visible from the UK, for radiant association.

Radiant RA/Dec at peak and daily drift from the IMO Meteor Shower Calendar /
Working List (https://www.imo.net/resources/calendar/), rounded; good to a
couple of degrees, which is within the matching tolerance. Peak dates are the
usual solar-longitude dates; they shift by a day year to year. Each shower is
matched only inside its activity window, and the radiant is moved by the
drift for the days from peak.

A meteor belongs to a shower if its great circle, extended backwards, passes
near the radiant and the radiant lies behind the trail (not beside or ahead
of it), with an extra allowance for the radiant's size.
"""

from datetime import date

import numpy

from . import astro
from .lens import altaz_to_vec, angle_between

# code, name, start (m, d), peak (m, d), end (m, d), RA, Dec, dRA/day, dDec/day, V km/s
SHOWERS = [
    ('QUA', 'Quadrantids',          (12, 28), (1, 3),   (1, 12),  230, +49, 0.80, -0.20, 41),
    ('LYR', 'Lyrids',               (4, 14),  (4, 22),  (4, 30),  271, +34, 1.10, 0.00, 49),
    ('ETA', 'eta Aquariids',        (4, 19),  (5, 6),   (5, 28),  338, -1,  0.90, 0.40, 66),
    ('SDA', 'Southern delta Aquariids', (7, 12), (7, 30), (8, 23), 340, -16, 0.80, 0.20, 41),
    ('CAP', 'alpha Capricornids',   (7, 3),   (7, 30),  (8, 15),  307, -10, 0.90, 0.30, 23),
    ('PER', 'Perseids',             (7, 17),  (8, 12),  (8, 24),  48,  +58, 1.35, 0.12, 59),
    ('KCG', 'kappa Cygnids',        (8, 3),   (8, 17),  (8, 25),  286, +59, 0.20, 0.00, 23),
    ('DRA', 'Draconids',            (10, 6),  (10, 8),  (10, 10), 262, +54, 0.00, 0.00, 20),
    ('STA', 'Southern Taurids',     (9, 10),  (10, 10), (11, 20), 32,  +9,  0.80, 0.30, 27),
    ('ORI', 'Orionids',             (10, 2),  (10, 21), (11, 7),  95,  +16, 0.70, 0.10, 66),
    ('NTA', 'Northern Taurids',     (10, 20), (11, 12), (12, 10), 58,  +22, 0.80, 0.20, 29),
    ('LEO', 'Leonids',              (11, 6),  (11, 17), (11, 30), 152, +22, 0.70, -0.42, 71),
    ('AMO', 'alpha Monocerotids',   (11, 15), (11, 21), (11, 25), 117, +1,  0.00, 0.00, 65),
    ('NOO', 'November Orionids',    (11, 13), (11, 28), (12, 6),  91,  +16, 0.80, 0.00, 44),
    ('MON', 'December Monocerotids', (12, 5), (12, 9),  (12, 20), 100, +8,  0.90, -0.10, 41),
    ('HYD', 'sigma Hydrids',        (12, 3),  (12, 9),  (12, 20), 125, +2,  0.70, -0.20, 58),
    ('GEM', 'Geminids',             (12, 4),  (12, 14), (12, 20), 112, +33, 1.00, -0.15, 35),
    ('COM', 'Comae Berenicids',     (12, 12), (12, 16), (12, 23), 175, +18, 0.80, -0.20, 65),
    ('DLM', 'December Leonis Minorids', (12, 5), (12, 19), (2, 4), 161, +30, 0.70, -0.30, 64),
    ('URS', 'Ursids',               (12, 17), (12, 22), (12, 26), 217, +76, 0.00, 0.00, 33),
]

# backward great circle within this of the radiant (radiant size + our
# pointing and fitting errors)
TOLERANCE_DEG = 5.0


def active(d):
    """Showers active on date d, with their radiant (RA, Dec) for that date.
    Windows may span the new year (start month/day after end)."""
    out = []
    for code, name, start, peak, end, ra, dec, dra, ddec, v in SHOWERS:
        for y in (d.year - 1, d.year):
            s = date(y, *start)
            e = date(y + 1 if end < start else y, *end)
            if s <= d <= e:
                p = date(y + 1 if peak < start else y, *peak)
                dp = (d - p).days
                out.append({'code': code, 'name': name, 'ra': (ra + dra * dp) % 360.0, 'dec': dec + ddec * dp,
                            'days_from_peak': dp, 'velocity': v})
                break
    return out


def associate(v1, v2, dt, lat, lon, showers=None):
    """Return the best-matching active shower for a trail with end vectors v1,
    v2 (ENU unit vectors) at time dt, or None.

    The trail's great circle must pass within TOLERANCE_DEG of the radiant,
    and the radiant must lie behind the trail: beyond the nearer end, never
    between or past the far end."""
    if showers is None:
        showers = active(dt.date())
    normal = numpy.cross(v1, v2)
    nn = numpy.linalg.norm(normal)
    if nn < 1e-6:
        return None
    normal /= nn
    length = float(angle_between(v1, v2))
    best = None
    for s in showers:
        alt, az = astro.radec_to_altaz(s['ra'], s['dec'], dt, lat, lon)
        if alt < numpy.radians(-5):
            continue  # radiant well below the horizon: no meteors from it
        r = altaz_to_vec(alt, az)
        off_circle = abs(float(numpy.degrees(numpy.arcsin(numpy.clip(numpy.dot(r, normal), -1, 1)))))
        if off_circle > TOLERANCE_DEG:
            continue
        d1 = float(angle_between(r, v1))
        d2 = float(angle_between(r, v2))
        near, far = min(d1, d2), max(d1, d2)
        # behind the trail: the far end is about one trail length beyond the near end
        if far - near < 0.7 * length:
            continue
        # a meteor can't be more than ~90 deg from its radiant
        if far > numpy.radians(100):
            continue
        cand = dict(s, off_deg=round(off_circle, 2), radiant_alt=round(float(numpy.degrees(alt)), 1),
                    start_dist_deg=round(float(numpy.degrees(near)), 1))
        if best is None or cand['off_deg'] < best['off_deg']:
            best = cand
    return best
