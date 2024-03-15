import traceback
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from interactive_tool.modules.app_visualizer import AppWindowO3DVisualizer
from lib.utils.misc import fix_seeds

fix_seeds(42)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--path', default=None, type=Path)
    parser.add_argument('--width', default=1920, type=int)
    parser.add_argument('--height', default=1080, type=int)
    # hyperparameters for the gui functions
    parser.add_argument('--texthreshold', default=0.075, type=float)
    parser.add_argument('--neighbour_threshold', default=0.075, type=float)
    # to load latest label by default
    parser.add_argument('--load_annot', default=False, type=bool)
    # model settings
    parser.add_argument('--pretrained_path', default='./checkpoints/closenet.pth', type=Path)
    parser.add_argument('--palette_path', default='assets/demo/color_palette.npy', type=Path)

    args = parser.parse_args()

    # Load the color palette
    palette = np.load(args.palette_path, allow_pickle=True)

    try:
        app = AppWindowO3DVisualizer(
            window_name='CloSeT',
            width=args.width,
            height=args.height,
            palette=palette,
            pretrained_path=args.pretrained_path,
            load_annot=args.load_annot,
            texthreshold=args.texthreshold,
            neighbour_threshold=args.neighbour_threshold,
        )
        app.run()
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()
