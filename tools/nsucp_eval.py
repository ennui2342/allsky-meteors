#!/usr/bin/env python3
"""Evaluate the scorer's single-frame rules (and optionally GMN's CNN) on
NightSkyUCP, and pull real meteor brightness profiles out of it.

  nsucp_eval.py DATASET_DIR OUT_DIR [--model hyper_model.tflite] [--limit N]

DATASET_DIR is the unzipped cr_dataset/ of NightSkyUCP (TU Ilmenau / AllSky7,
doi:10.6084/m9.figshare.16451625, CC BY-NC-SA 4.0): 10k meteor and 10k hard
non-meteor detections from AllSky7 video cameras, each with a "sum image"
(frames summed, first frame subtracted). A sum image is what a long exposure
of the event records, so it stands in for our new-light difference `d`.

For every sum image: find the longest streak with detect.detect, measure it
with detect.measure and apply scorer.shape_veto and scorer.score -- the rules
that need no neighbouring frames (the continuation and satellite tests can't
run on single events). Writes OUT_DIR/results.csv and prints rates per class
and per labelled non-meteor subclass (birds, clouds, light, plane, rain,
trees).

Profiles: for meteors found with peak SNR >= 15, the brightness along the
trail (extended 8 px past the detected ends, trimmed at 5 % of peak) resampled to 64 points and normalised to peak 1, saved to
OUT_DIR/profiles.npy for the injection test (scorer inject --profiles).

Run with a Python that has numpy, opencv, scipy (and ai-edge-litert for
--model); the meteors/ directory must be on sys.path.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allsky_meteors import detect, scorer   # noqa: E402

PROFILE_POINTS = 64
# the sum images are tight crops around the event, so the "frame too busy"
# skip (a whole-frame guard against cloud and dawn) doesn't apply
detect.BUSY_FRACTION = 1.0


def subclasses(root):
    out = {}
    for p in (root / 'raw/labels/subclasses').glob('*.txt'):
        for line in p.read_text().split():
            out[Path(line).stem] = p.stem
    return out


def evaluate(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = detect.gray(img)
    d = g - numpy.median(g)
    mask = numpy.ones(g.shape, bool)
    idx = numpy.flatnonzero(mask)
    # the sums are near noiseless; a 2 DN floor keeps JPEG ringing out of
    # the streak width (lower floors make most trails too fat to pass)
    sigma = max(detect.robust_sigma(d, idx), 2.0)
    streaks, info = detect.detect(d, mask, sigma)
    if not streaks:
        return {'found': False, 'd': d}
    s = max(streaks, key=lambda s: s['len'])
    f = detect.measure(d, img, numpy.zeros_like(img), s, sigma)
    # stand-ins for the sky-geometry features score() wants
    f['length_deg'] = s['len'] * 0.14
    f['alt_mid_deg'] = 45.0
    cls, reason = scorer.shape_veto(f)
    conf, _ = scorer.score(f)
    if cls:
        conf = 0
    else:
        cls = 'meteor' if conf >= scorer.METEOR_CONF else 'possible' if conf >= scorer.POSSIBLE_CONF else 'junk'
    # the profile runs 8 px past the detected ends, so faint ends below the
    # detection threshold are kept; trimmed to where it's lit (main())
    u = (s['p2'] - s['p1']) / max(float(numpy.hypot(*(s['p2'] - s['p1']))), 1e-6)
    lim = numpy.array([d.shape[1] - 1, d.shape[0] - 1], float)
    e1, e2 = numpy.clip(s['p1'] - 8 * u, 0, lim), numpy.clip(s['p2'] + 8 * u, 0, lim)
    prof, _ = detect.profile(d, e1, e2)
    return {'found': True, 'd': d, 's': s, 'f': f, 'cls': cls, 'reason': reason or '', 'conf': conf,
            'sigma': sigma, 'prof': prof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--model')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sub = subclasses(args.dataset)

    it = None
    if args.model:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
        it = Interpreter(model_path=args.model)
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        _, mh, mw, _ = inp['shape']

    rows, profiles = [], []
    for label in ('meteor', 'non_meteor'):
        files = sorted((args.dataset / 'sum_images' / label).glob('*.jpg'))
        if args.limit:
            files = files[:args.limit]
        for n, p in enumerate(files):
            r = evaluate(p)
            if r is None:
                continue
            row = {'label': label, 'subclass': sub.get(p.stem, ''), 'id': p.stem, 'found': int(r['found']),
                   'cls': r.get('cls', 'none'), 'reason': r.get('reason', ''), 'conf': r.get('conf', 0)}
            if r['found']:
                f = r['f']
                for k in ('length_px', 'fwhm_px', 'peak_snr', 'taper', 'coverage', 'ripple_rel', 'ripple',
                          'beads', 'bead_density', 'dashed', 'side_ratio', 'local_activity', 'colour_var', 'elong'):
                    row[k] = f[k]
            if it is not None:
                d = r['d']
                if r['found']:
                    crop = scorer.ml_crop(d, r['s']['p1'], r['s']['p2'])
                else:
                    crop = numpy.clip(d, 0, 255).astype(numpy.uint8)
                x = cv2.resize(crop, (mw, mh)).astype(numpy.float32)[None, :, :, None] / 255.0
                it.set_tensor(inp['index'], x)
                it.invoke()
                row['gmn'] = round(float(it.get_tensor(outp['index']).ravel()[0]), 3)
            rows.append(row)
            if label == 'meteor' and r['found'] and r['f']['peak_snr'] >= 15 and r['f']['length_px'] >= 25:
                prof = numpy.convolve(numpy.clip(r['prof'], 0, None), numpy.ones(3) / 3, mode='same')
                lit = numpy.nonzero(prof > 0.05 * prof.max())[0]
                prof = prof[lit[0]:lit[-1] + 1] if len(lit) else prof
                if prof.max() > 0 and len(prof) >= 10:
                    t = numpy.linspace(0, 1, len(prof))
                    profiles.append(numpy.interp(numpy.linspace(0, 1, PROFILE_POINTS), t, prof / prof.max()))
            if n % 1000 == 0:
                print(f'{label} {n}/{len(files)}', file=sys.stderr, flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(args.out / 'results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    numpy.save(args.out / 'profiles.npy', numpy.array(profiles, numpy.float32))
    report(rows)
    print(f'{len(profiles)} meteor profiles -> {args.out / "profiles.npy"}')


def report(rows):
    def tally(rs):
        n = len(rs)
        c = {}
        for r in rs:
            c[r['cls']] = c.get(r['cls'], 0) + 1
        return n, c
    for label in ('meteor', 'non_meteor'):
        rs = [r for r in rows if r['label'] == label]
        n, c = tally(rs)
        print(f'{label}: n={n} ' + ' '.join(f'{k}={v} ({v / n:.0%})' for k, v in sorted(c.items())))
        why = {}
        for r in rs:
            if r['reason']:
                k = r['reason'].split('(')[0].strip()
                why[k] = why.get(k, 0) + 1
        print('   vetoes:', dict(sorted(why.items(), key=lambda x: -x[1])))
        if 'gmn' in rs[0]:
            g = numpy.array([r['gmn'] for r in rs])
            print('   gmn: ' + ' '.join(f'>{t}={(g > t).mean():.0%}' for t in (0.5, 0.8, 0.9, 0.95)))
    for sc in sorted({r['subclass'] for r in rows if r['subclass']}):
        n, c = tally([r for r in rows if r['subclass'] == sc])
        print(f'  non-meteor {sc:7s} n={n:3d} ' + ' '.join(f'{k}={v}' for k, v in sorted(c.items())))


if __name__ == '__main__':
    main()
