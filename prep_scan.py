import trimesh
import numpy as np
import pickle as pkl
import torch
import cv2
import argparse
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Pointclouds, Meshes, packed_to_list
from pytorch3d.ops import knn_points
from pytorch3d.ops import estimate_pointcloud_normals
import smplx


label2class = np.array(
    [
        'Hat',
        'Body',
        'Shirt',
        'TShirt',
        'Vest',
        'Coat',
        'Dress',
        'Skirt',
        'Pants',
        'ShortPants',
        'Shoes',
        'Hoodies',
        'Hair',
        'Swimwear',
        'Underwear',
        'Scarf',
        'Jumpsuits',
        'Jacket',
    ]
)


def convert_to_textureVertex(texture, input_data) -> TexturesVertex:
    verts_colors_packed = torch.zeros_like(input_data.verts_packed())
    verts_colors_packed[
        input_data.faces_packed()
    ] = texture.faces_verts_textures_packed().cuda()  # (*)
    return TexturesVertex(packed_to_list(verts_colors_packed, input_data.num_verts_per_mesh()))


def load_mesh(
    input_path, texture_path=None, device='cuda'
) -> tuple[Meshes, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    verts, faces, aux = load_obj(input_path)
    # * aux doesn't read normal info
    try:
        trimesh_mesh = trimesh.load(input_path, process=False, maintain_order=True)
        vert_normals = torch.from_numpy(trimesh_mesh.vertex_normals.astype(np.float32))
    except Exception:
        pc = Pointclouds(points=[verts]).to(device)
        vert_normals = estimate_pointcloud_normals(pc, neighborhood_size=50).float()[0]

    if vert_normals.shape[0] != verts.shape[0]:
        pc = Pointclouds(points=[verts]).to(device)
        vert_normals = estimate_pointcloud_normals(pc, neighborhood_size=50).float()[0]

    texture = None
    if texture_path is not None:
        # Load image
        texture_image = cv2.imread(str(texture_path), cv2.COLOR_BGR2RGB)
        # It's important to convert image to float
        texture_image = torch.from_numpy(texture_image.astype(np.float32) / 255)

        # Extract representation needed to create Textures object
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        texture_image = texture_image[None, ...]  # (1, H, W, 3)

        texture = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    scan_mesh = Meshes(
        verts=[verts], faces=[faces.verts_idx], textures=texture, verts_normals=[vert_normals]
    ).to(device)
    verts_tex = convert_to_textureVertex(texture, scan_mesh)
    col_val = verts_tex.verts_features_list()[0].to(device).unsqueeze(0)
    col_val = torch.floor(col_val * 255.0)
    norms = scan_mesh.verts_normals_list()[0].unsqueeze(0)
    mesh_verts, mesh_faces = scan_mesh.get_mesh_verts_faces(0)
    return scan_mesh, mesh_verts, mesh_faces, col_val, norms


def humanbody_data(src_verts, mesh_verts) -> np.ndarray:
    template_mesh = trimesh.load('./assets/template_mesh.obj', process=False)
    template_verts = np.asarray(template_mesh.vertices)
    # load smpl+d and smpl vertices from registraion data
    dst_verts = mesh_verts.unsqueeze(0)
    dist, idx, nn = knn_points(dst_verts, src_verts)
    canon_pose_locations_smpl = template_verts[idx.detach().cpu().numpy()[0, :, 0]]

    return canon_pose_locations_smpl


def create_smpl(smpl_file, bm_dir_path, device='cuda'):
    if '.pkl' in smpl_file:
        smpl_data = pkl.load(open(smpl_file, 'rb'))
    else:
        smpl_data = dict(np.load(open(smpl_file, 'rb'), allow_pickle=True))

    gender = 'neutral'
    if 'gender' in smpl_data.keys():
        gender = smpl_data['gender']
    body_model = smplx.create(
        model_path=bm_dir_path, gender=gender, model_type='smpl', batch_size=1, num_betas=10
    )
    body_model = body_model.to(device=device)

    betas = torch.from_numpy(smpl_data['betas'].reshape(1, 10).astype(np.float32)).to(device=device)
    body_pose = torch.from_numpy(smpl_data['body_pose'].astype(np.float32).reshape(1, 69)).to(
        device=device
    )
    global_orient = torch.from_numpy(
        smpl_data['global_orient'].reshape(1, 3).astype(np.float32)
    ).to(device=device)
    full_pose = torch.cat([global_orient, body_pose], dim=-1)[0].detach().cpu().numpy()
    smpl_trans = smpl_data['transl']

    if 'scale' in smpl_data.keys():
        smpl_scale = smpl_data['scale']
    else:
        smpl_scale = 1.0
    body_model_output = body_model.forward(
        betas=betas, body_pose=body_pose, global_orient=global_orient
    )
    smpl_verts = body_model_output.vertices

    smpl_verts = smpl_verts * torch.from_numpy(smpl_scale.astype(np.float32)).to(device=device)
    smpl_verts = smpl_verts + torch.from_numpy(smpl_trans.astype(np.float32)).to(device=device)
    return smpl_verts, smpl_scale, smpl_trans, full_pose, betas


def main(scan_obj, scan_tex, smpl_file, save_path, garment_class, bm_dir_path, device='cuda'):
    # * load scan mesh using pytorch3d
    _, mesh_verts, mesh_faces, col_val, norms = load_mesh(scan_obj, scan_tex, device=device)

    # * load registration and create canonical points
    smpl_verts, smpl_scale, smpl_trans, full_pose, betas = create_smpl(
        smpl_file, bm_dir_path, device
    )

    canon_pose_locations_smpl = humanbody_data(smpl_verts, mesh_verts)

    # * normalize the scan
    scan_mesh = trimesh.Trimesh(
        vertices=mesh_verts.detach().cpu().numpy(),
        faces=mesh_faces.detach().cpu().numpy(),
        maintain_order=True,
        process=False,
    )
    scan_mesh = trimesh.Trimesh(vertices=smpl_verts[0].detach().cpu().numpy())
    total_size = (scan_mesh.bounds[1] - scan_mesh.bounds[0]).max()
    centers = (scan_mesh.bounds[1] + scan_mesh.bounds[0]) / 2

    scan_mesh.apply_translation(-centers)
    scan_mesh.apply_scale(1 / total_size)

    # * garment class names to labels
    garment_class = (label2class[:, None] == garment_class).argmax(axis=0)
    garments = np.zeros(18)
    garments[garment_class] = 1
    np.savez(
        save_path,
        points=scan_mesh.vertices,
        normals=norms.detach().cpu().numpy(),
        colors=col_val.detach().cpu().numpy()[0],
        faces=scan_mesh.faces,
        scale=(1.0 / total_size),
        pose=full_pose,
        betas=betas.detach().cpu().numpy()[0],
        trans=smpl_trans,
        canon_pose=canon_pose_locations_smpl,
        garments=garments,
    )


# ! Example call:
# * python prepare_data.py \
# * --scan_obj $SOME_PATH/0000.obj \        -- This file represents the scan mesh;
# * --scan_tex $SOME_PATH/material0.jpeg \  -- This file represents UV texture map for the scan mesh;
# * --smpl_file $SOME_PATH/0000_smpl.pkl \  -- This file represents SMPL mesh fit and is expected to contain keys: betas, body_pose, global_orient, transl, scale; 
# * --save_path $SOME_PATH/0000.npz \
# * --bm_dir_path $SMPL_PATH/models \       -- This folder represents the SMPL body model directory; See https://github.com/vchoutas/smplx for more details on setting up SMPL model;
# * --garment_class TShirt Pants Body Hair Shoes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--scan_obj', help='Path to scan obj file', type=str)
    parser.add_argument('-t', '--scan_tex', help='Path to scan texture file', type=str)
    parser.add_argument('-r', '--smpl_file', help='Path to SMPL registration file', type=str)
    parser.add_argument('-g', '--garment_class', nargs='+', default=[])
    parser.add_argument('-s', '--save_path', type=str)
    parser.add_argument('-b', '--bm_dir_path', help='Path to SMPL body model directory', type=str)
    args = parser.parse_args()

    main(
        args.scan_obj,
        args.scan_tex,
        args.smpl_file,
        args.save_path,
        args.garment_class,
        args.bm_dir_path,
        device='cuda',
    )
    print('data saved for scan:', args.scan_obj, 'at:', args.save_path)
