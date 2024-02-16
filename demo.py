import pickle as pkl
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from lib.closed import SingleScanDataset
from lib.utils.config import load_model
from lib.utils.misc import exists, fix_seeds, mkdir, save_image
from lib.utils.viz import render_p3d

if __name__ == '__main__':
    parser = ArgumentParser(description='CloSeNet inference demo')
    parser.add_argument(
        '--model',
        type=str,
        default='./pretrained/closenet.pth',
        help='Path to the model checkpoint',
    )
    parser.add_argument(
        '--scan_path',
        type=str,
        default='./assets/demo/demo_scan.npz',
        help='Path to input .npz file',
    )
    parser.add_argument(
        '--output', type=str, default='./out/demo_scan_out', help='Path to output directory'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--render', action='store_true', default=False, help='Render the output')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for dataloader')
    args = parser.parse_args()

    assert exists(args.model), f'{args.model} does not exist'
    assert exists(args.scan_path), f'{args.scan_path} does not exist'
    assert args.model.endswith('.pth'), 'Model should be a .pth file'
    assert args.scan_path.endswith('.npz'), 'Input scan should be a .npz file'

    save_dir = mkdir(args.output)

    # * Fix seeds for reproducibility
    fix_seeds(args.seed)

    # * Load the model
    model = load_model(args.model, device=args.device)
    model.eval()

    # * NOTE: We create SingleScanDataset instance to load the scan data,
    # * and create a dataloader to iterate over the scan data,
    # * because scan can be a large collection of points (eg. ~100k points),
    # * thus would trigger CudaOutOfMemoryError if passed through the model in a single forward pass.
    scan = SingleScanDataset(args.scan_path)
    loader = scan.get_loader(batch_size=2048, num_workers=args.n_workers)
    n_points = scan.points.shape[0]
    # * Run inference
    with torch.no_grad():
        outputs = defaultdict(list)
        prog_bar = tqdm(loader, desc=f'Running inference on {n_points} points')
        for batch in prog_bar:
            batch = batch.to(args.device)
            pred = model(batch)

            # * Aggregate outputs
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    outputs[k].append(v.cpu().squeeze())
                else:
                    outputs[k].append(v)

    # * Save outputs
    out_path = save_dir / 'outputs.pkl'
    print(f'Saving outputs {out_path}')
    pkl.dump(outputs, open(out_path, 'wb'))

    # * Render outputs
    if args.render:
        print('Rendering outputs...')
        color_palette = (
            torch.from_numpy(np.load('./assets/demo/color_palette.npy')).to(args.device).float()
        )
        points = torch.vstack(outputs['points'])[..., :3].to(args.device)
        colors = torch.vstack(outputs['points'])[..., 3:6].to(args.device)
        pred_lbl_colors = torch.cat(outputs['labels'], axis=0).to(args.device)
        inv_indices = torch.cat(outputs['idx'], axis=0).sort()[1].to(args.device)
        shape_faces = inv_indices[pred.faces.long()].to(args.device)
        input_scan_img = np.hstack(
            render_p3d(
                vertices=points,
                faces=shape_faces,
                colors=colors.to(args.device),
                azimuths=[0, 90, 180, 270],
                resolution=128,
                input_type='mesh',
            )[..., :3]
            * 255.0
        )
        inferred_scan_img = np.hstack(
            render_p3d(
                vertices=points,
                faces=shape_faces,
                colors=color_palette[pred_lbl_colors],
                azimuths=[0, 90, 180, 270],
                resolution=128,
                input_type='mesh',
            )[..., :3]
            * 255.0
        )
        img_path = save_dir / 'result.png'
        print(f'Saving rendered images {img_path}')
        save_image(np.vstack([input_scan_img, inferred_scan_img]), img_path)
