from typing import Union

import numpy as np
import open3d as o3d
from lib.utils.misc import labels_to_colors
import os
from datetime import datetime
import glob
from pathlib import Path


def to_vector3i(arr: Union[np.ndarray, list]):
    if not isinstance(arr, o3d.utility.Vector3dVector):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return o3d.utility.Vector3iVector(arr)
    else:
        return arr


def to_vector3d(arr: Union[np.ndarray, list]):
    if not isinstance(arr, o3d.utility.Vector3dVector):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return o3d.utility.Vector3dVector(arr)
    else:
        return arr


def set_model_grad(model, mode='segm_dec'):
    """Set the model to training mode for the specified components"""
    model.eval()

    # Initially assert that the model is not trainable
    for param in model.parameters():
        param.requires_grad = False
    # Verify that the model is not trainable
    assert model.training is False, 'Model is not trainable'

    modes = mode.split('+')

    if 'segm_dec' in modes:
        for param in model.segm_dec.parameters():
            param.requires_grad = True
    if 'segm_dec_last' in modes:
        for param in model.segm_dec.layers[5].parameters():
            param.requires_grad = True
    if 'mlp_pc_enc' or 'mlp_pc_enc_last' in modes:
        for param in model.pc_enc.lin_global.parameters():
            param.requires_grad = True
    if 'edge_conv_feat3' or 'edge_conv_feat3_last' in modes:
        for param in model.pc_enc.convs[-1].parameters():
            param.requires_grad = True


def relabel(mesh, new_label, selected_vertices, palette):
    new_labels = np.repeat(new_label, len(selected_vertices))
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertex_colors[selected_vertices] = labels_to_colors(new_labels, palette)
    mesh.vertex_colors = to_vector3d(vertex_colors)

    return mesh


def get_timestamp() -> str:
    return f'{datetime.now().strftime("%y%m%d-%H%M%S")}'


def assign_ids(datapath, selected_path):
    # Use glob to find all files ending with .npz
    npz_files = glob.glob(f'{datapath}/*.npz')
    # Sort the list of files by name
    npz_files.sort()
    # Assign ids to the files
    id_dict = {}
    for i, npz_file in enumerate(npz_files):
        id_dict[i] = npz_file
    inverse_id_dict = {v: k for k, v in id_dict.items()}
    curr_scan_id = inverse_id_dict[selected_path]
    return id_dict, curr_scan_id


def find_save(data_path, save_name='*.npy'):
    save_path = Path(data_path).with_name(save_name)
    label_file = sorted(glob.glob(str(save_path)), key=os.path.getmtime)
    if label_file == []:
        return False
    data_labels = np.load(label_file[-1], allow_pickle=True)
    return data_labels


def label_by_texture(
    data_points, data_colors, data_labels, only_background=True, tthreshold=0.075, threshold=0.075
):
    texture = to_vector3d(data_colors)
    if only_background:
        ref_vertices = np.where(data_labels == -1)[0]
    else:
        ref_vertices = range(len(data_points))
    data_labels_orig = data_labels.copy()
    for ref_vertex in ref_vertices:
        ref_texture = texture[ref_vertex]
        # find all indices of the vertices with similar texture color
        condition = np.linalg.norm(texture - ref_texture, axis=1) < tthreshold
        changed_vertices_all = np.where(condition)[0]
        ref_point = np.asarray(data_points[ref_vertex]).reshape(1, -1)
        # change all vertices within a threshold distance
        point_with_texture = data_points[changed_vertices_all]
        distances = np.linalg.norm(point_with_texture - ref_point, axis=1)
        changed_vertices = changed_vertices_all[distances < threshold]
        sel_labels = data_labels_orig[changed_vertices]
        values, counts = np.unique(sel_labels, return_counts=True)
        if len(values) == 0:
            continue
        new_label = values[np.argmax(counts)]
        data_labels[changed_vertices] = new_label
    return data_labels


def load_mesh(
    data_path,
    color_palette,
    load_annot=False,
    assign_by_texture=False,
    only_background=True,
    alpha=0.3,
):
    """Load the mesh vertices, faces, and colors from the data_path"""
    data = dict(np.load(data_path, allow_pickle=True))
    path = Path(data_path)
    scan_name = path.stem

    data_points = data['points'] if len(data['points'].shape) == 2 else data['points'][0, :, :]
    data_colors = data['colors'] if len(data['colors'].shape) == 2 else data['colors'][0, :, :]
    if data_colors.any() > 1:
        data_colors = data_colors / 255.0

    data_faces = data['faces'] if len(data['faces'].shape) == 2 else data['faces'][0, :, :]

    if 'pred_labels' in data.keys():
        data_labels = data['pred_labels']
        data['labels'] = data['pred_labels'].copy()
    else:
        data_labels = data['labels'] if len(data['labels'].shape) == 1 else data['labels'][0, :]
    if load_annot:
        # load the latest annotation (.npy file) if present
        data_labels = find_save(save_name=f'*{scan_name}.npy')
        if data_labels:
            data['labels'] = data_labels
    elif assign_by_texture:
        print('Assigning labels wrt texture')
        print('Only background:', only_background)
        texture_labels = label_by_texture(
            data_points, data_colors, data_labels, only_background=only_background
        )
        data['labels'] = texture_labels

    mesh = o3d.geometry.TriangleMesh(
        vertices=to_vector3d(data_points), triangles=to_vector3i(data_faces)
    )

    mesh.vertex_colors = blend_colors(mesh, data, color_palette, alpha=alpha)
    return mesh, data


def blend_colors(mesh, data, palette, selected_vertices=None, alpha=0.3):
    """Blend the colors of the mesh vertices with the segmentation label colors"""
    if selected_vertices is None:
        selected_vertices = np.arange(len(data['labels']))  # all vertices
    data_colors = data['colors'] if len(data['colors'].shape) == 2 else data['colors'][0, :, :]
    if np.any(data_colors > 1):
        data_colors = data_colors / 255.0
    seg_colors = labels_to_colors(data['labels'], palette)

    mesh_vertex_colors = np.asarray(mesh.vertex_colors)
    if mesh_vertex_colors.shape[0] == len(data['labels']):
        blend = alpha * data_colors[selected_vertices] + (1 - alpha) * seg_colors[selected_vertices]
        mesh_vertex_colors[selected_vertices] = to_vector3d(blend)
    else:
        mesh_vertex_colors = to_vector3d(alpha * data_colors + (1 - alpha) * seg_colors)
    return mesh_vertex_colors


def save_file(path_to_save, scan_name, data_labels):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # save into data_path
    save_name = f'{get_timestamp()}_{scan_name}.npy'
    save_path = path_to_save.joinpath(save_name)
    np.save(save_path, data_labels)
    return save_path
