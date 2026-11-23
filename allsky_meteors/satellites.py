"""Known-satellite veto: was a sunlit satellite crossing along this streak
during the exposure?

Orbits come from indi-allsky's own `tle_data` table (Celestrak groups it
refreshes daily: ~10,800 satellites, including Starlink), or, where that is
empty (a SyncAPI receiver doesn't fetch TLEs), straight from Celestrak's
active group, cached for a day under the results directory. Every satellite is
propagated at once with sgp4's vectorised SatrecArray over the exposure
window; TEME -> Earth-fixed uses sidereal time only (no polar motion or
nutation, ~0.05 deg), which is far inside the 1.5 deg match tolerance.
"""

import logging
import math
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy
from sgp4.api import Satrec, SatrecArray

from . import astro, db

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6378.137
WGS84_F = 1 / 298.257223563
MATCH_DEG = 1.5      # both streak ends within this of the satellite's path
OVERLAP_DEG = 3.0    # slack along the path (timestamp and download jitter)
CELESTRAK_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
CACHE_HOURS = 24     # Celestrak asks clients not to fetch the same data more often than it updates
RETRY_HOURS = 3      # after a failed fetch


def _celestrak(cache):
    """(title, line1, line2) from Celestrak, via a file cached for a day."""
    cache = Path(cache)
    attempt = cache.with_suffix('.attempt')
    stale = not cache.exists() or time.time() - cache.stat().st_mtime > CACHE_HOURS * 3600
    # one try per RETRY_HOURS: Celestrak blocks clients that keep re-fetching
    # (it answers 403 within its update interval)
    due = not attempt.exists() or time.time() - attempt.stat().st_mtime > RETRY_HOURS * 3600
    if stale and due:
        cache.parent.mkdir(parents=True, exist_ok=True)
        attempt.touch()
        try:
            req = urllib.request.Request(CELESTRAK_URL, headers={'User-Agent': 'allsky-meteors'})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode('ascii', 'replace')
            if text.count('\n1 ') > 100:
                cache.write_text(text)
        except OSError as e:
            logger.warning('Celestrak fetch failed (%s); using the cached TLEs', e)
    if not cache.exists():
        return []
    lines = [ln.rstrip() for ln in cache.read_text().splitlines() if ln.strip()]
    return [(lines[i], lines[i + 1], lines[i + 2]) for i in range(0, len(lines) - 2, 3)
            if lines[i + 1].startswith('1 ') and lines[i + 2].startswith('2 ')]


def load_tles(cache):
    try:
        rows = [(r['title'], r['line1'], r['line2']) for r in db.query('select title, line1, line2 from tle_data')]
    except Exception:  # no tle_data table
        rows = []
    if not rows:
        rows = _celestrak(cache)
    names, sats, seen = [], [], set()
    for title, l1, l2 in rows:
        num = l1[2:7]
        if num in seen:
            continue
        seen.add(num)
        try:
            sats.append(Satrec.twoline2rv(l1, l2))
        except Exception:  # malformed TLE
            continue
        names.append(title.strip())
    logger.info('loaded %d satellites', len(names))
    return names, SatrecArray(sats)


def _observer_ecef(lat_deg, lon_deg, h_km):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    n = EARTH_RADIUS_KM / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return numpy.array([(n + h_km) * math.cos(lat) * math.cos(lon),
                        (n + h_km) * math.cos(lat) * math.sin(lon),
                        (n * (1 - e2) + h_km) * math.sin(lat)])


class SatelliteVeto:
    def __init__(self, tle_cache, lat, lon, elevation_m):
        self.names, self.array = load_tles(tle_cache)
        self.lat, self.lon = lat, lon
        self.obs = _observer_ecef(lat, lon, elevation_m / 1000.0)
        la, lo = math.radians(lat), math.radians(lon)
        self.enu = numpy.array([
            [-math.sin(lo), math.cos(lo), 0.0],
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
        ])

    def tracks(self, start, end, samples=8):
        """Directions (n_sat, samples, 3, ENU unit vectors) of satellites that
        are above the horizon and sunlit at some point in [start, end]; plus
        their indices."""
        times = [start + (end - start) * i / (samples - 1) for i in range(samples)]
        jd = numpy.array([astro.julian_date(t) for t in times])
        whole = numpy.floor(jd - 0.5) + 0.5
        err, r_teme, _ = self.array.sgp4(whole, jd - whole)
        ok = (err == 0).all(axis=1)
        dirs = numpy.empty_like(r_teme)
        for i, t in enumerate(times):
            g = astro.gmst_rad(t)
            rot = numpy.array([[math.cos(g), math.sin(g), 0], [-math.sin(g), math.cos(g), 0], [0, 0, 1]])
            r_ecef = r_teme[:, i, :] @ rot.T
            d = (r_ecef - self.obs) @ self.enu.T
            dirs[:, i, :] = d / numpy.linalg.norm(d, axis=1, keepdims=True)
        up = (dirs[:, :, 2] > 0).any(axis=1)
        # sunlit at mid-exposure: outside Earth's cylindrical shadow
        mid = samples // 2
        sun = astro.sun_vec_eq(times[mid])
        r = r_teme[:, mid, :]
        proj = r @ sun
        perp = numpy.linalg.norm(r - proj[:, None] * sun, axis=1)
        sunlit = (proj > 0) | (perp > EARTH_RADIUS_KM)
        idx = numpy.nonzero(ok & up & sunlit)[0]
        return dirs[idx], idx

    def match(self, v1, v2, frame_end, exposure_s):
        """Best satellite whose path during the exposure runs along the streak
        from v1 to v2 (ENU unit vectors), as (name, max offset deg), or None."""
        start = frame_end - timedelta(seconds=exposure_s + 2.0)
        end = frame_end + timedelta(seconds=0.5)
        dirs, idx = self.tracks(start, end)
        if not len(idx):
            return None
        p0, p1 = dirs[:, 0, :], dirs[:, -1, :]
        normal = numpy.cross(p0, p1)
        span = numpy.linalg.norm(normal, axis=1)
        keep = span > math.sin(math.radians(0.2))   # moved at least 0.2 deg
        if not keep.any():
            return None
        p0, p1, normal, idx = p0[keep], p1[keep], normal[keep] / span[keep, None], idx[keep]
        arc = numpy.arctan2(numpy.linalg.norm(numpy.cross(p0, p1), axis=1), numpy.sum(p0 * p1, axis=1))
        offs, along = [], []
        for v in (v1, v2):
            offs.append(numpy.abs(numpy.degrees(numpy.arcsin(numpy.clip(normal @ v, -1, 1)))))
            along.append(numpy.degrees(numpy.arctan2(numpy.sum(normal * numpy.cross(p0, v), axis=1), p0 @ v)))
        off = numpy.maximum(*offs)
        lo, hi = numpy.minimum(*along), numpy.maximum(*along)
        arc_deg = numpy.degrees(arc)
        overlap = (hi > -OVERLAP_DEG) & (lo < arc_deg + OVERLAP_DEG)
        good = (off < MATCH_DEG) & overlap
        if not good.any():
            return None
        best = numpy.argmin(numpy.where(good, off, numpy.inf))
        return self.names[idx[best]], round(float(off[best]), 2)
