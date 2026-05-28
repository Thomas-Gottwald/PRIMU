#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from glob import glob
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
import torchvision.transforms.functional as TF
from utils.read_pfm import read_pfm

class CameraInfo:
    def __init__(self, uid, R, T, FovY, FovX, image, image_path, image_name, width, height, pose,
                 depth=None):
        self.uid = uid
        self.R = R
        self.T = T
        self.FovY = FovY
        self.FovX = FovX
        self.image = image
        self.image_path = image_path
        self.image_name = image_name
        self.width = width
        self.height = height
        self.pose = pose
        self.depth = depth# for LF depth

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    depth_scale: float = 1.0


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    def try_open_image_variants(base_path):
        """Try opening the image with common file extension variants."""
        for ext in ["", ".jpg", ".JPG", ".png"]:
            path = base_path if ext == "" else os.path.splitext(base_path)[0] + ext
            if os.path.exists(path):
                return np.array(Image.open(path))
        return None
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        pose = np.pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" :
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              pose=pose)
        
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, split_file=None, llffhold=8, colmap_folder="sparse", init_ply_file="points3D.ply"):
    try:
        cameras_extrinsic_file = os.path.join(path, f"{colmap_folder}/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"{colmap_folder}/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"{colmap_folder}/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"{colmap_folder}/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    
    reading_dir = "images" if images == None else images
    
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if os.path.isfile(split_file):
        with open(split_file, "r") as file:
            split_text = file.read()
        train_split_text, test_split_text = split_text.split(os.linesep+"test:"+os.linesep)
        train_split_info = train_split_text.split(os.linesep)
        test_split_info = test_split_text.split(os.linesep)

        train_cam_infos = [c for c in cam_infos if c.image_name in train_split_info]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_split_info]
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0] 
 
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"{colmap_folder}/0/{init_ply_file}")
    bin_path = os.path.join(path, f"{colmap_folder}/0/points3D.bin")
    txt_path = os.path.join(path, f"{colmap_folder}/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path) 
    return scene_info

# LF data reader from variational-3dgs https://github.com/csrqli/variational-3dgs/blob/360fcccecfdf9e80a69a75d2428c74d4d0aab247/scene/dataset_readers.py#L196
def readLFSceneInfo(path, images, eval, llffhold=8, use360=False):
    scene_name = os.path.basename(path)

    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    #train/test split from CF-NeRF https://github.com/poetrywanderer/CF-NeRF/blob/66918a9748c137e1c0242c12be7aa6efa39ece06/run_nerf_uncertainty_NF.py#L750
    #   depth_scale are from https://github.com/csrqli/variational-3dgs/blob/main/scene/dataset_readers.py#L212
    if scene_name == 'basket':
        i_train = list(np.arange(43,50,2))
        i_val = list(np.arange(42,50,2))
        depth_scale = 8# variational-3dgs uses here 1 / 8 but this dose not fit!

    elif scene_name == 'africa':
        i_train = list(np.arange(5,14,2))
        i_val = list(np.arange(6,14,2))
        depth_scale = 4

    elif scene_name == 'statue':
        i_train = list(np.arange(67,76,2))
        i_val = list(np.arange(68,76,2))
        depth_scale = 1.25

    elif scene_name == 'torch':
        i_train = list(np.arange(8,17,2))
        i_val = list(np.arange(9,17,2))
        depth_scale = 15

    ####
    # use 360 degree training views
    if use360:
        i_train = []
        for idx in range(len(cam_infos)):
            if idx not in i_val:
                i_train.append(idx)
    ###

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in i_train]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in i_val]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    all_depth = []
    depth_files = glob(os.path.join(path, f'depth_*.npy'))
    depth_files = sorted(depth_files)
    for i in range(4): 
        depth = np.ascontiguousarray(np.load(depth_files[i]))
        depth = TF.to_tensor(depth).cuda()
        test_cam_infos[i].depth = depth

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           depth_scale=depth_scale)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]
            use_fov = True
        elif "fx" in contents and "fy" in contents:
            focal_length_x = contents["fx"]
            focal_length_y = contents["fy"]
            use_fov = False
        else:
            raise ValueError()

        frames = contents["frames"]
        # search for npy or pfm depth maps
        frame0 = contents["frames"][0]
        depth_file_npy = os.path.join(path, frame0["file_path"] + "_depth.npy")
        depth_file_pfm = os.path.join(path, frame0["file_path"] + "_depth.pfm")
        if os.path.isfile(depth_file_npy):
            found_depth = "npy"
        elif os.path.isfile(depth_file_pfm):
            found_depth = "pfm"
        else:
            found_depth = ""
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            pose = np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            if use_fov:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                FovY = focal2fov(focal_length_y, image.size[1])
                FovX = focal2fov(focal_length_x, image.size[0])
            
            depth = None
            if found_depth == "npy":
                # if npy depth files are found load them
                depth_file = os.path.join(path, frame["file_path"] + "_depth.npy")
                depth = np.ascontiguousarray(np.load(depth_file))
                depth = np.nan_to_num(depth, nan=np.nan, posinf=np.nan, neginf=np.nan)# set +-inf to nan
                nan_idc = np.where(depth!=depth)# find nan positions for later
                depth = TF.to_tensor(depth).cuda()

                # fill nan values as mean of surrounding not nan values
                #  (if there are no surrounding not nan values throw an error)
                # nan_mask = depth!=depth
                for i in range(len(nan_idc[0])):
                    start0 = nan_idc[0][i]-1 if nan_idc[0][i]!=0 else 0
                    stop0 = nan_idc[0][i]+2 if nan_idc[0][i]!=depth.shape[1]-1 else depth.shape[1]
                    start1 = nan_idc[1][i]-1 if nan_idc[1][i]!=0 else 0
                    stop1 = nan_idc[1][i]+2 if nan_idc[1][i]!=depth.shape[2]-1 else depth.shape[2]
                    depth[:,nan_idc[0][i],nan_idc[1][i]] = torch.nanmean(depth[:,start0:stop0,start1:stop1])
                assert not torch.any(depth.isnan()), f"Found to many NaN/Inf entries in ground truth depth of {frame['file_path']}!"
            elif found_depth == "pfm":
                # if pfm depth files are found load them
                depth_file = os.path.join(path, frame["file_path"] + "_depth.pfm")
                depth = np.ascontiguousarray(read_pfm(depth_file)[0])
                depth[depth==0.0] = np.nan
                depth = TF.to_tensor(depth).cuda()
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                             image=image, image_path=image_path, image_name=image_name,
                             width=image.size[0], height=image.size[1],
                             pose=pose, depth=depth)) 
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", init_ply_file="points3D.ply"): 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension) 

    with open(os.path.join(path, "transforms_train.json")) as json_file:
        contents_train = json.load(json_file)
    depth_scale = contents_train["depth_scale"] if "depth_scale" in contents_train else 1.0

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, init_ply_file)
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           depth_scale=depth_scale)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "LF": readLFSceneInfo,
    "Blender" : readNerfSyntheticInfo
}