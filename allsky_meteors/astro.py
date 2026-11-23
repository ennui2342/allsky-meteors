"""Small astronomy helpers: sidereal time, RA/Dec -> alt/az, the Sun's
direction. Low precision (~0.1 deg) is plenty for matching streaks that are
several degrees long; no ephemeris files or network needed."""

import math
from datetime import datetime, timezone

import numpy


def julian_date(dt):
    return dt.astimezone(timezone.utc).timestamp() / 86400.0 + 2440587.5


def gmst_rad(dt):
    """Greenwich mean sidereal time (IAU 1982), radians."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t ** 3 / 38710000.0
    return math.radians(gmst % 360.0)


def radec_to_altaz(ra_deg, dec_deg, dt, lat_deg, lon_deg):
    """J2000-ish RA/Dec (no precession: < 0.5 deg error to 2030) -> alt, az radians."""
    lst = gmst_rad(dt) + math.radians(lon_deg)
    ha = lst - numpy.radians(ra_deg)
    dec = numpy.radians(dec_deg)
    lat = math.radians(lat_deg)
    alt = numpy.arcsin(numpy.sin(dec) * math.sin(lat) + numpy.cos(dec) * math.cos(lat) * numpy.cos(ha))
    az = numpy.arctan2(-numpy.cos(dec) * numpy.sin(ha),
                       numpy.sin(dec) * math.cos(lat) - numpy.cos(dec) * math.sin(lat) * numpy.cos(ha))
    return alt, numpy.mod(az, 2 * numpy.pi)


def sun_radec(dt):
    """Sun's apparent RA/Dec in degrees (Astronomical Almanac low-precision
    formula, ~0.01 deg)."""
    n = julian_date(dt) - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    return ra, dec


def sun_vec_eq(dt):
    """Unit vector to the Sun in the equatorial frame (x to RA 0, z to the pole)."""
    ra, dec = (math.radians(v) for v in sun_radec(dt))
    return numpy.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def now_utc():
    return datetime.now(timezone.utc)
