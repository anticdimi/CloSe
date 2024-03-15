from typing import Union
import cv2

import numpy as np
import torch

from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds


def create_renderer(
    input_type: str,
    azimuth: list[int],
    image_size: int = 512,
    elevation: float = 5.0,
    up=((0, 1, 0),),
    center=((0, 0, 0),),
    dist: float = 2.0,
    device: torch.device = 'cpu',
):
    # Initialize a camera
    # With world coordinates +Y up, +X left and +Z in
    R, T = look_at_view_transform(dist=dist, elev=elevation, azim=azimuth, up=up, at=center)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    if input_type == 'pointcloud':
        # Define the settings for rasterization and shading
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, radius=0.005, points_per_pixel=10, bin_size=128
        )

        # Create a renderer by composing a rasterizer and a shader
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=[1, 1, 1]),
        )
    elif input_type == 'mesh':
        # Define the settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1
        )

        # Place point lights
        lights = PointLights(
            ambient_color=[[0.7, 0.7, 0.7]],
            diffuse_color=[[0.2, 0.2, 0.2]],
            specular_color=[[0.1, 0.1, 0.1]],
            location=[[0.0, 5.0, 0.0]],
            device=device,
        )

        # Create a Phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )
    else:
        raise RuntimeError(f'Unsupported input type {input_type}')

    return renderer


def get_center(input_data: Union[Meshes, Pointclouds]) -> torch.Tensor:
    if isinstance(input_data, Meshes):
        vertices_list = input_data.verts_list()[0]
    elif isinstance(input_data, Pointclouds):
        vertices_list = input_data.points_list()[0]
    else:
        RuntimeError(f'Unsupported input type {type(input_data)}')

    center = vertices_list.mean(0)
    return torch.unsqueeze(center, 0)


def render(
    input_data: Union[Meshes, Pointclouds],
    azimuths: list[int],
    image_size: int,
    elevation: int = 10,
    dist: float = 2.0,
    camera_up: tuple[float] = ((0, 1, 0),),
    input_type: str = 'mesh',
) -> np.ndarray:
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Get center location (to control camera orientation)
    center = get_center(input_data)

    # Create renderer
    renderer = create_renderer(
        input_type=input_type,
        azimuth=azimuths,
        image_size=image_size,
        elevation=elevation,
        dist=dist,
        center=center,
        device=device,
        up=camera_up,
    )

    # Perform batch rendering
    images = renderer(input_data.extend(len(azimuths)))

    return images.cpu().numpy()


def render_p3d(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: np.ndarray,
    azimuths: list,
    input_type: str = 'mesh',
    resolution: int = 512,
) -> np.ndarray:
    device = vertices.device
    if input_type == 'mesh':
        input_mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex([colors])).to(
            device
        )
    else:
        input_mesh = Pointclouds(points=[vertices], features=[colors]).to(device)
    return render(
        input_data=input_mesh,
        azimuths=azimuths,
        image_size=resolution,
        camera_up=((0, 1, 0),),
        dist=1.0,
        elevation=10,
        input_type=input_type,
    )


def render_wrapper(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: np.ndarray,
    device: str,
    azimuths: list,
    input_type: str = 'mesh',
) -> None:
    colors = torch.from_numpy(colors).float() if isinstance(colors, np.ndarray) else colors.float()
    if input_type == 'mesh':
        input_mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex([colors])).to(
            device
        )
    else:
        input_mesh = Pointclouds(points=[vertices], features=[colors]).to(device)
    return render(
        input_data=input_mesh,
        azimuths=azimuths,
        image_size=2048,
        camera_up=((0, 1, 0),),
        dist=1.0,
        elevation=10,
        input_type=input_type,
    )


def save_views(
    points,
    shape_faces,
    texture,
    render_device,
    azimuths,
    input_type,
    pred_lbl_colors,
    gt_lbl_colors,
    save_path,
    rank,
    scan_id,
):
    input_img = render_wrapper(
        points,
        shape_faces,
        texture,
        render_device,
        azimuths,
        input_type=input_type,
    )
    # Segmentation predictions
    pred_img = render_wrapper(
        points,
        shape_faces,
        pred_lbl_colors,
        render_device,
        azimuths,
        input_type=input_type,
    )
    # Segmentation ground truth
    gt_img = render_wrapper(
        points,
        shape_faces,
        gt_lbl_colors,
        render_device,
        azimuths,
        input_type=input_type,
    )

    gt = np.hstack([cv2.cvtColor(gt_img[i] * 255, cv2.COLOR_RGB2BGR) for i in range(len(gt_img))])
    pred = np.hstack(
        [cv2.cvtColor(pred_img[i] * 255, cv2.COLOR_RGB2BGR) for i in range(len(pred_img))]
    )
    input = np.hstack([input_img[i] * 255 for i in range(len(input_img))])

    result = np.vstack([input, pred, gt])
    cv2.imwrite(str(save_path / f'{rank:03d}_{scan_id}.png'), result)

    # Save input views
    for i, azim in enumerate(azimuths):
        cv2.imwrite(
            str(save_path / f'{rank:03d}_{scan_id}_{azim:03d}_input.png'), input_img[i] * 255
        )

    # Save GT views
    for i, azim in enumerate(azimuths):
        cv2.imwrite(
            str(save_path / f'{rank:03d}_{scan_id}_{azim:03d}_gt.png'),
            cv2.cvtColor(gt_img[i] * 255, cv2.COLOR_RGB2BGR),
        )

    # Save pred views
    for i, azim in enumerate(azimuths):
        cv2.imwrite(
            str(save_path / f'{rank:03d}_{scan_id}_{azim:03d}_pred.png'),
            cv2.cvtColor(pred_img[i] * 255, cv2.COLOR_RGB2BGR),
        )
