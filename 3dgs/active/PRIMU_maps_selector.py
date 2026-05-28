import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from typing import List

from PRIMU_create_feature_maps import compute_gaussian_representations, render_feature_maps
from fused_ssim import fused_ssim_map as ssim
from lpipsPyTorch import lpips_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss


def create_selection_visualization_plots(iteration, scene, train_cameras, test_cameras, candidate_cameras,
                                         candidate_feature_maps, candidate_scores_dict, feature_map,
                                         selected_idx):

    for i,fmaps in enumerate(tqdm(candidate_feature_maps, desc="Plotting selection visualizations")):
        image_name = candidate_scores_dict["image"][i]
        score = candidate_scores_dict["score"][i]
        score_l1 = candidate_scores_dict["l1"][i]
        score_psnr = candidate_scores_dict["psnr"][i]
        score_ssim = candidate_scores_dict["ssim"][i]
        score_lpips = candidate_scores_dict["lpips"][i]

        fig = plt.figure(figsize=(15,10))

        axes00 = fig.add_subplot(2,2,1)
        axes00.imshow(fmaps["gt_rgb"].transpose(1,2,0))
        axes00.set_title(f'ground truth [l1={score_l1:.4f}, psnr={score_psnr:.4f}]')
        axes00.axis("off")

        axes01 = fig.add_subplot(2,2,2)
        axes01.imshow(fmaps["pred_rgb"].transpose(1,2,0))
        axes01.set_title(f'render [ssim={score_ssim:.4f}, lpips={score_lpips:.4f}]')
        axes01.axis("off")

        axes10 = fig.add_subplot(2,2,3, projection="3d")
        elev = 30
        azim = 90
        if "flower" in scene.model_path:
            elev = 20
            azim = -60
        elif "room" in scene.model_path:
            elev = 30
            azim = 130
        plot_camp_positions(axes10, current_idx=i, selected_idx=selected_idx, train_cameras=train_cameras, test_cameras=test_cameras,
                            candidate_cameras=candidate_cameras, elev=elev, azim=azim)
        axes10.set_title(f"score={score:.4f}")

        axes11 = fig.add_subplot(2,2,4)
        im1 = axes11.imshow(fmaps[feature_map], cmap="turbo", interpolation="nearest")
        axes11.set_title(f"{feature_map}")
        axes11.axis("off")
        fig.colorbar(im1, ax=axes11, orientation="horizontal")

        ####
        save_file_npz = Path(scene.model_path) / "PRIMU_avs" / "npz" / f"iteration_{iteration}" / f"{image_name}.npz"
        save_file_npz.parent.mkdir(parents=True, exist_ok=True)

        np.savez(save_file_npz, gt_rgb=fmaps["gt_rgb"], pred_rgb=fmaps["pred_rgb"], feature_map=fmaps[feature_map])
        ####

        plt.tight_layout()
        save_file = Path(scene.model_path) / "PRIMU_avs" / "selection_plots" / f"iteration_{iteration}" / f"{image_name}.png"
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)


def plot_camp_positions(ax, current_idx, selected_idx, train_cameras, test_cameras, candidate_cameras,
                        elev=30, azim=90, roll=0):
    candidate_pos = np.zeros(shape=(len(candidate_cameras),3))
    candidate_rot = np.zeros(shape=(len(candidate_cameras),3))
    for i,candidate_view in enumerate(candidate_cameras):        
        R = candidate_view.R
        T = candidate_view.T
        position = -R @ T[:, None]
        candidate_pos[i] = position.reshape(-1)
        candidate_rot[i] = R @ np.array([0.0, 0.0, 1.0])
    current_pos = candidate_pos[current_idx:current_idx+1]
    current_rot = candidate_rot[current_idx:current_idx+1]
    selected_pos = candidate_pos[selected_idx:selected_idx+1]
    selected_rot = candidate_rot[selected_idx:selected_idx+1]
    candidate_pos = candidate_pos[np.arange(len(candidate_cameras)) != current_idx]
    candidate_rot = candidate_rot[np.arange(len(candidate_cameras)) != current_idx]
    ax.scatter(candidate_pos[:,0],candidate_pos[:,1],candidate_pos[:,2], label="candidate", c="gray", alpha=0.5, marker="x")
    ax.quiver(candidate_pos[:,0],candidate_pos[:,1],candidate_pos[:,2],
              candidate_rot[:,0],candidate_rot[:,1],candidate_rot[:,2], length=0.5, colors="gray", alpha=0.5)
    
    test_pos = np.zeros(shape=(len(test_cameras),3))
    test_rot = np.zeros(shape=(len(test_cameras),3))
    for i,test_view in enumerate(test_cameras):        
        R = test_view.R
        T = test_view.T
        position = -R @ T[:, None]
        test_pos[i] = position.reshape(-1)
        test_rot[i] = R @ np.array([0.0, 0.0, 1.0])
    ax.scatter(test_pos[:,0],test_pos[:,1],test_pos[:,2], label="test", c="tab:blue", alpha=0.7, marker="o")
    ax.quiver(test_pos[:,0],test_pos[:,1],test_pos[:,2],
              test_rot[:,0],test_rot[:,1],test_rot[:,2], length=0.5, colors="tab:blue", alpha=0.7)
    
    train_pos = np.zeros(shape=(len(train_cameras),3))
    train_rot = np.zeros(shape=(len(train_cameras),3))
    for i,train_view in enumerate(train_cameras):        
        R = train_view.R
        T = train_view.T
        position = -R @ T[:, None]
        train_pos[i] = position.reshape(-1)
        train_rot[i] = R @ np.array([0.0, 0.0, 1.0])
    ax.scatter(train_pos[:,0],train_pos[:,1],train_pos[:,2], label="train", c="tab:red", marker="s")
    ax.quiver(train_pos[:,0],train_pos[:,1],train_pos[:,2],
              train_rot[:,0],train_rot[:,1],train_rot[:,2], length=1.0, colors="tab:red")
    
    selected_label = "selected (current)" if selected_idx == current_idx else "selected"
    ax.scatter(selected_pos[:,0],selected_pos[:,1],selected_pos[:,2], label=selected_label, c="green", marker="*")
    ax.quiver(selected_pos[:,0],selected_pos[:,1],selected_pos[:,2],
              selected_rot[:,0],selected_rot[:,1],selected_rot[:,2], length=1.0, colors="green")

    if selected_idx != current_idx:
        ax.scatter(current_pos[:,0],current_pos[:,1],current_pos[:,2], label="current", c="black", marker="D")
        ax.quiver(current_pos[:,0],current_pos[:,1],current_pos[:,2],
                current_rot[:,0],current_rot[:,1],current_rot[:,2], length=1.0, colors="black")
        
    ######
    npz_cam_pos_file = Path("/home/gottwald/code_release/PRIMU/output_active/PRIMU/mipNeRF360/garden/zz_cam_pos.npz")
    if not npz_cam_pos_file.is_file():
        train_img_names = []
        for train_view in train_cameras:
            train_img_names.append(train_view.image_name)
        train_img_names = np.array(train_img_names)

        test_img_names = []
        for test_view in train_cameras:
            test_img_names.append(test_view.image_name)
        test_img_names = np.array(test_img_names)

        candidate_img_names = []
        candidate_pos = np.zeros(shape=(len(candidate_cameras),3))
        candidate_rot = np.zeros(shape=(len(candidate_cameras),3))
        for i,candidate_view in enumerate(candidate_cameras):        
            R = candidate_view.R
            T = candidate_view.T
            position = -R @ T[:, None]
            candidate_pos[i] = position.reshape(-1)
            candidate_rot[i] = R @ np.array([0.0, 0.0, 1.0])
            candidate_img_names.append(candidate_view.image_name)
        
        np.savez(npz_cam_pos_file, img_names=train_img_names, test_img_names=test_img_names, candidate_img_names=candidate_img_names,
                 train_pos=train_pos, train_rot=train_rot, test_pos=test_pos, test_rot=test_rot,
                 candidate_pos=candidate_pos, candidate_rot=candidate_rot)
    exit(0)
    ######

    ax.legend()
    ax.view_init(elev=elev, azim=azim, roll=roll)


class PRIMUMapsSelector(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.lpips = lpips_func("cuda", net_type='vgg')

    def forward(self, x):
        return x
    
    def nbvs(self, gaussians, scene, num_views, dataset, pipe, background, iteration, *args, **kwargs) -> List[int]:
        with torch.no_grad():
            print("Computing Gaussian representations...")
            gaussian_representations = compute_gaussian_representations(gaussians, scene, dataset, pipe, self.args.sh_deg, self.args.kappa)

            candidate_views = list(deepcopy(scene.get_candidate_set()))
            candidate_cameras = scene.getCandidateCameras()
            train_cameras = scene.getTrainCameras()
            test_cameras = scene.getTestCameras()

            candidate_feature_maps = render_feature_maps(dataset, gaussians, pipe, gaussian_representations, candidate_cameras, self.args.sh_deg)

            candidate_scores_dict = {
                "image": [],
                "score": [],
                "l1": [],
                "psnr": [],
                "ssim": [],
                "lpips": [],
            }
            for i,candidate_view in enumerate(tqdm(candidate_cameras, desc="Computing feature map scores and quality metrics")):
                # candidate view file name
                candidate_scores_dict["image"].append(candidate_view.image_name)

                # score for active view selection
                fmaps = candidate_feature_maps[i]
                if "visibility" in self.args.feature_map:
                    candidate_scores_dict["score"].append(-np.mean(fmaps[self.args.feature_map]))
                else:
                    candidate_scores_dict["score"].append(np.mean(fmaps[self.args.feature_map]))

                # gs quality
                pred_img = torch.tensor(fmaps["pred_rgb"]).to("cuda")
                gt_img = torch.tensor(fmaps["gt_rgb"]).to("cuda")
                score_l1 = l1_loss(pred_img, gt_img).mean().item()
                candidate_scores_dict["l1"].append(score_l1)
                score_psnr = psnr(pred_img, gt_img).mean().item()
                candidate_scores_dict["psnr"].append(score_psnr)
                score_ssim = ssim(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean().item()
                candidate_scores_dict["ssim"].append(score_ssim)
                self.lpips.to(gt_img.device)
                score_lpips = self.lpips(pred_img, gt_img).mean().item()
                candidate_scores_dict["lpips"].append(score_lpips)

            df_scores = pd.DataFrame.from_dict(candidate_scores_dict)
            score_path = Path(scene.model_path) / "PRIMU_avs" / "scores" / f"iteration_{iteration}_scores.csv"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            df_scores.to_csv(score_path)

            if num_views == 1:
                # Return list of the index of the candidate view with the highest error score
                best_idx = int(np.argmax(candidate_scores_dict["score"]))

                if self.args.plot_selection:
                    create_selection_visualization_plots(iteration, scene, train_cameras, test_cameras, candidate_cameras,
                                                         candidate_feature_maps, candidate_scores_dict, feature_map=self.args.feature_map,
                                                         selected_idx=best_idx)

                return [candidate_views[best_idx]]
            else:
                # Multi-view selection is not yet implemented
                raise NotImplementedError("Multi-view selection is not yet implemented")
