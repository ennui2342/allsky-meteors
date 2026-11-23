#!/usr/bin/env python3
"""Score crops with the Global Meteor Network's meteor/artefact CNN.

  gmn_score.py MODEL.tflite CROPS_DIR [CROPS_DIR ...]

MODEL is one of RMS's share/*.tflite (github.com/CroatianMeteorNetwork/RMS,
GPL-3.0; hyper_model.tflite is RMS's default, 60x60 input). CROPS_DIR holds
<class>/<id>.png greyscale crops as written by `scorer mlcrops`, or any tree
of PNG/JPG crops; the class is the parent directory name. Preprocessing
follows RMS/MLFilter.py: greyscale, resized to the model's input, /255.
Prints per-class score distributions and writes scores.csv next to the
first CROPS_DIR.

Needs a TensorFlow Lite runtime: pip install ai-edge-litert numpy opencv-python-headless
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


def main():
    model, dirs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    it = Interpreter(model_path=model)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    _, h, w, _ = inp['shape']
    rows = []
    for d in dirs:
        for p in sorted(d.rglob('*')):
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            x = cv2.resize(img, (w, h)).astype(numpy.float32)[None, :, :, None] / 255.0
            it.set_tensor(inp['index'], x)
            it.invoke()
            score = float(it.get_tensor(out['index']).ravel()[0])
            rows.append((p.parent.name, str(p), score))
    by = {}
    for cls, _, s in rows:
        by.setdefault(cls, []).append(s)
    print(f'{Path(model).name}: class  n  >0.5  median  (scores)')
    for cls, ss in sorted(by.items()):
        ss = numpy.array(ss)
        tail = ' '.join(f'{v:.2f}' for v in sorted(ss)[:12]) if len(ss) <= 12 else ''
        print(f'  {cls:10s} {len(ss):5d} {int((ss > 0.5).sum()):5d}  {numpy.median(ss):.2f}  {tail}')
    with open(dirs[0] / f'scores_{Path(model).stem}.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('class', 'path', 'score')] + rows)


if __name__ == '__main__':
    main()
