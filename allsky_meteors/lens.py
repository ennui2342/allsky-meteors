"""Pixel <-> sky mapping for the saved (display-orientation) frames.

Same model as indi-allsky's lens solver and VirtualSky overlay
(indi_allsky/lens_solver/projection.py): an equisolid fisheye

    r = (diameter / 2) * sin(zenith_angle / 2) / sin(45 deg)

centred at (width/2 + offset_x, height/2 - offset_y), azimuth measured from
north through east, rotated by `azimuth_deg`. With mirror=False east is on the
left of the frame, as on this camera.

Parameters come from indi-allsky's config (VIRTUALSKY.IMAGE_CIRCLE_DIAMETER,
VIRTUALSKY.OFFSET_X/Y, LENS_AZIMUTH), which the web UI's lens solver writes
once it has fitted a clear night. Until then LENS_AZIMUTH is 0 and the
scorer uses the value measured from the Moon on 2026-11-20 (7 deg).
"""

import math

import numpy

SIN45 = math.sin(math.radians(45.0))


class Lens:
    def __init__(self, width, height, diameter, offset_x, offset_y, azimuth_deg, mirror=False):
        self.width = width
        self.height = height
        self.radius = diameter / 2.0
        self.cx = width / 2.0 + offset_x
        self.cy = height / 2.0 - offset_y
        self.azimuth = math.radians(azimuth_deg)
        self.sign = 1.0 if mirror else -1.0

    def to_xy(self, alt, az):
        """alt, az in radians (scalars or arrays) -> pixel x, y."""
        theta = numpy.pi / 2.0 - numpy.asarray(alt)
        r = self.radius * numpy.sin(theta / 2.0) / SIN45
        psi = numpy.asarray(az) - self.azimuth
        return self.cx + self.sign * r * numpy.sin(psi), self.cy - r * numpy.cos(psi)

    def to_altaz(self, x, y):
        """pixel x, y -> alt, az in radians."""
        dx = numpy.asarray(x, dtype=float) - self.cx
        dy = numpy.asarray(y, dtype=float) - self.cy
        r = numpy.hypot(dx, dy)
        s = numpy.clip(r * SIN45 / self.radius, 0.0, 1.0)
        theta = 2.0 * numpy.arcsin(s)
        psi = numpy.arctan2(self.sign * dx, -dy)
        return numpy.pi / 2.0 - theta, numpy.mod(psi + self.azimuth, 2 * numpy.pi)

    def to_vec(self, x, y):
        """pixel -> unit vector (east, north, up)."""
        return altaz_to_vec(*self.to_altaz(x, y))

    def px_per_deg(self, x, y):
        """Local plate scale (radial) at a pixel, for converting widths."""
        alt, _ = self.to_altaz(x, y)
        theta = numpy.pi / 2.0 - alt
        return self.radius / SIN45 * 0.5 * numpy.cos(theta / 2.0) * numpy.pi / 180.0


def altaz_to_vec(alt, az):
    alt = numpy.asarray(alt)
    az = numpy.asarray(az)
    return numpy.stack([numpy.cos(alt) * numpy.sin(az), numpy.cos(alt) * numpy.cos(az), numpy.sin(alt)], axis=-1)


def vec_to_altaz(v):
    v = numpy.asarray(v)
    return numpy.arcsin(numpy.clip(v[..., 2], -1, 1)), numpy.mod(numpy.arctan2(v[..., 0], v[..., 1]), 2 * numpy.pi)


def angle_between(a, b):
    """Angle in radians between unit vectors (broadcasts)."""
    return numpy.arctan2(numpy.linalg.norm(numpy.cross(a, b), axis=-1), numpy.sum(a * b, axis=-1))


def from_indi_config(cfg, width, height, default_azimuth=7.0):
    vs = cfg.get('VIRTUALSKY', {}) or {}
    diameter = vs.get('IMAGE_CIRCLE_DIAMETER') or cfg.get('LENS_IMAGE_CIRCLE') or 937
    azimuth = cfg.get('LENS_AZIMUTH') or 0.0
    if not azimuth:
        azimuth = default_azimuth  # not solved yet: Moon-based measurement
    return Lens(width, height, float(diameter), float(vs.get('OFFSET_X', 0)), float(vs.get('OFFSET_Y', 0)),
                float(azimuth))
