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
import random
import json
from glob import glob
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, load_type="iteration", shuffle=True, resolution_scales=[1.0]): 
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # print("Loading scene from {}".format(args.model_path))

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.eval and not os.path.isfile(args.split_file) and len(glob(os.path.join(args.source_path, f'depth_*.npy'))) > 0:
            scene_info = sceneLoadTypeCallbacks["LF"](args.source_path, args.images, args.eval, use360=args.useLF360)
        elif os.path.exists(os.path.join(args.source_path, args.colmap_folder)):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, split_file=args.split_file,
                                                          colmap_folder=args.colmap_folder, init_ply_file=args.init_ply_file)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, init_ply_file=args.init_ply_file) 
        else:
            assert False, "Could not recognize scene type!"
        self.depth_scale = scene_info.depth_scale

        self.train_cameras_info = scene_info.train_cameras
        self.test_cameras_info = scene_info.test_cameras

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # For active view selection
        # train idxs are indices into the train cameras list
        # For active view selection, this gets overwritten, by default it is all views
        num_views = len(scene_info.train_cameras)
        self.all_train_set = set(range(num_views)) # Keeps track of all training views to select from in active view selection
        self.train_idxs = list(range(num_views))

        if shuffle:
            # Todo: May need to make this deterministic for comparing multiple methods
            # random.Random(args.seed).shuffle(self.train_idxs)  # Multi-res consistent random shuffling
            # train_idxs shuffling has no effect when using active view selection
            # random.shuffle(self.train_idxs) 
            # Do not need to shuffle train cameras as we retrieve them shuffeled with getTrainCameras
            # random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(" Number of training cameras: ", len(self.train_cameras[resolution_scale]))
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(" Number of test cameras: ", len(self.test_cameras[resolution_scale]))


        if self.loaded_iter != -2:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     load_type + "_" + str(self.loaded_iter),
                                                     "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent) 

        self.candidate_views_filter = None

        # holdout view for uq regressor
        self.holdout_idxs = []

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_error(self, iteration, name="error"):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/{}_{}".format(name,iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        filted_train_camers = [self.train_cameras[scale][i] for i in self.train_idxs]
        return filted_train_camers

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    # Candidate set for active view selection
    def get_candidate_set(self):
        candidate_set = sorted(list(self.all_train_set - set(self.train_idxs) - set(self.holdout_idxs)))
        if self.candidate_views_filter is not None: # Can apply additional filtering here, currently not in use
            candidate_set = list(filter(self.candidate_views_filter, candidate_set))
        return candidate_set

    def getCandidateCameras(self, scale=1.0):
        candidate_set = list(self.get_candidate_set())
        filted_train_camers = [self.train_cameras[scale][i] for i in candidate_set]
        return filted_train_camers
    
    def getHoldoutCameras(self, scale=1.0):
        filted_train_camers = [self.train_cameras[scale][i] for i in self.holdout_idxs]
        return filted_train_camers