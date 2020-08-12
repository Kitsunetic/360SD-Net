import argparse
import math
from pathlib import Path

import numpy as np


def disp2depth(disp: np.ndarray, angle: np.ndarray, angle2: np.ndarray):
    h, w = disp.shape
    mask0 = disp == 0
    maskn0 = disp > 0

    de_pred = np.zeros((h, w))
    de_pred[maskn0] = (angle[maskn0] / np.tan(disp[maskn0] / 180 * math.pi)) \
                      + angle2[maskn0]
    de_pred[mask0] = 0
    return de_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=0.2, type=float, help='The input baseline')
    parser.add_argument('path', type=str, help='The input folder path')
    args = parser.parse_args()

    input_dir = Path(args.path)
    out_dir = input_dir.parent / (input_dir.name + '_depth')
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = args.baseline

    # angle array for calculation
    angle = np.zeros((512, 1024))
    angle2 = np.zeros((512, 1024))
    pi_h = math.pi / 512
    for i in range(1024):
        for j in range(512):
            theta_T = math.pi - ((j + 0.5) * pi_h)
            angle[j, i] = baseline * math.sin(theta_T)
            angle2[j, i] = baseline * math.cos(theta_T)

    for file in input_dir.glob('*.npy'):
        print(file.name)
        disp = np.load(str(file))
        depth = disp2depth(disp, angle, angle2)
        np.save(str(out_dir / file.name), depth)
