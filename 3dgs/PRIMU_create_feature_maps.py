import torch
from gaussian_renderer import gsplat_render as render, backproject
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, parse_args_with_cfg_args
import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt

from pathlib import Path
from einops import reduce
from utils.sh_utils import eval_sh

import jax
jax.config.update("jax_enable_x64", True)# just to not get a warning: we do not use jax
from s2fft.precompute_transforms.construct import spin_spherical_kernel
from s2fft.sampling.s2_samples import phis_equiang, thetas


class VisMapMode(IntEnum):
    MAX = 0
    MAXnoAlpha = 1
    SUM = 2
    SUMnoAlpha = 3
    PixCount = 4
    MEAN = 5
    MEANnoAlpha = 6

    def __str__(self):
        string = super().__str__()
        return string.split(".")[-1]


def render_feature_maps(dataset, gaussians, pipe, gaussian_representations, viewpoint_stack, sh_deg=0, depth_scale=1.0):
    maps = ["MAX", "MAXnoAlpha", "SUM", "SUMnoAlpha", "MEAN", "MEANnoAlpha"]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    background_maps = torch.tensor([0], dtype=torch.float32, device="cuda")

    feature_maps = []

    for i in tqdm(range(len(viewpoint_stack)), desc="Rendering feature maps"):
        viewpoint = viewpoint_stack[i]

        # render feature maps
        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, background,
                                render_mode="RGB+D" if viewpoint.depth is not None else "RGB")
            r_image = render_pkg["render"][:3]
            pred_rgb = torch.clamp(r_image, 0, 1).cpu()
            gt_rgb = viewpoint.original_image.cpu()
            if viewpoint.depth is not None:
                pred_depth = depth_scale * render_pkg["render"][3].cpu()
                gt_depth = viewpoint.depth.cpu()

            gs_fov_counter = gaussian_representations["fov_counter"].reshape(-1,1).to(dtype=torch.float32)
            render_pkg = render(viewpoint, gaussians, pipe, background_maps, override_color=gs_fov_counter)
            fov_counter = reduce(render_pkg["render"], "c h w -> h w", "mean").cpu()

            dir_pp = (gaussians.get_xyz - viewpoint.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            visibility_maps = []
            for j in range(6):
                gs_visibility = gaussian_representations["visibility"][...,j].to(dtype=torch.float32)
                shs_view = gs_visibility.reshape(-1,1,(sh_deg+1)**2)
                sh2rgb = eval_sh(sh_deg, shs_view, dir_pp_normalized)
                render_pkg = render(viewpoint, gaussians, pipe, background_maps, override_color=sh2rgb)
                visibility_maps.append(reduce(render_pkg["render"], "c h w -> h w", "mean").cpu())
            error_maps = []
            for j in range(6):
                gs_error = gaussian_representations["error"][...,j].to(dtype=torch.float32)
                shs_view = gs_error.reshape(-1,1,(sh_deg+1)**2)
                sh2rgb = eval_sh(sh_deg, shs_view, dir_pp_normalized)
                render_pkg = render(viewpoint, gaussians, pipe, background_maps, override_color=sh2rgb)
                error_maps.append(reduce(render_pkg["render"], "c h w -> h w", "mean").cpu())

        store = {
            "gt_rgb": gt_rgb.numpy(),
            "pred_rgb": pred_rgb.numpy(),
            "visibility_counter": fov_counter.numpy()
        }
        if viewpoint.depth is not None:
            store["gt_depth"] = gt_depth.numpy()
            store["pred_depth"] = pred_depth.numpy()
        for j in range(6):
            store[f"visibility_map_{maps[j]}_sh{sh_deg}"] = visibility_maps[j].cpu().numpy()
        for j in range(6):
            store[f"error_bp_map_{maps[j]}_sh{sh_deg}"] = error_maps[j].cpu().numpy()

        feature_maps.append(store)
    
    return feature_maps


def store_feature_maps(dataset, gaussians, pipe, gaussian_representations, viewpoint_stack, sh_deg=0, kappa=8, depth_scale=1.0, folder_npz="PRIMU/feature_maps",
                       plot=False, folder_plots="PRIMU/plots/feature_maps"):
    feature_maps_path = Path(dataset.model_path) / folder_npz

    feature_maps = render_feature_maps(dataset, gaussians, pipe, gaussian_representations, viewpoint_stack, sh_deg, depth_scale)

    for i in tqdm(range(len(viewpoint_stack)), desc="Saving feature maps"):
        store = feature_maps[i]
        
        feature_maps_file = feature_maps_path / f"{i:05d}_kappa{kappa:.2f}_SH{sh_deg}.npz"
        feature_maps_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feature_maps_file,
                 **store)
        
        if plot:
            plot_feature_maps(dataset, store, i, sh_deg, kappa, folder_plots)

    print(f"Rendering complete. Outputs saved to {str(feature_maps_path)}")


def plot_feature_maps(dataset, view_feature_maps, view_idx, sh_deg=0, kappa=8, folder_plots="PRIMU/plots/feature_maps"):
    """create plots of uncertainty feature maps for a single view"""

    gt_np = view_feature_maps["gt_rgb"].transpose(1, 2, 0)
    rendered_np = view_feature_maps["pred_rgb"].transpose(1, 2, 0)
    error_np = np.abs(rendered_np-gt_np).mean(axis=-1)

    if "gt_depth" in view_feature_maps:
        gt_depth_np = view_feature_maps["gt_depth"]
        rendered_depth_np = view_feature_maps["pred_depth"]
        error_depth_np = np.abs(rendered_depth_np-gt_depth_np)

    fov_counter = view_feature_maps["visibility_counter"]

    visibility_maps = []
    error_maps = []
    maps = ["MAX", "MAXnoAlpha", "SUM", "SUMnoAlpha", "MEAN", "MEANnoAlpha"]
    for j in range(6):
        visibility_maps.append(view_feature_maps[f"visibility_map_{maps[j]}_sh{sh_deg}"])
        error_maps.append(view_feature_maps[f"error_bp_map_{maps[j]}_sh{sh_deg}"])

    # Save plots of feature maps
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))

    axes[0,0].imshow(gt_np)
    axes[0,0].set_title('Ground Truth')
    axes[0,1].imshow(rendered_np)
    axes[0,1].set_title('Render')
    vmin = np.quantile(error_np, 0.01)
    vmax = np.quantile(error_np, 0.99)
    axes[0,2].imshow(error_np, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0,2].set_title('Error')
    if "gt_depth" in view_feature_maps:
        vmin = np.nanmin(gt_depth_np)
        vmax = np.nanmax(gt_depth_np)
        axes[0,3].imshow(gt_depth_np, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0,3].set_title('Ground Truth Depth')
        axes[0,4].imshow(rendered_depth_np, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0,4].set_title('Render Depth')
        vmin = np.quantile(error_depth_np, 0.01)
        vmax = np.quantile(error_depth_np, 0.99)
        axes[0,5].imshow(error_depth_np, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[0,5].set_title('Error Depth')
    for j in range(6):
        axes[0,j].axis('off')

    # FOV counter
    vmin = np.quantile(fov_counter, 0.01)
    vmax = np.quantile(fov_counter, 0.99)
    axes[1,0].imshow(fov_counter, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1,0].set_title('FOV Counter')
    for j in range(6):
        axes[1,j].axis('off')

    # Visibility
    for j in range(6):
        vmin = np.quantile(visibility_maps[j], 0.01)
        vmax = np.quantile(visibility_maps[j], 0.99)
        axes[2,j].imshow(visibility_maps[j], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[2,j].set_title(f'visibility ({maps[j]}, sh{sh_deg})')
        axes[2,j].axis('off')

    # Error
    for j in range(6):
        vmin = np.quantile(error_maps[j], 0.01)
        vmax = np.quantile(error_maps[j], 0.99)
        axes[3,j].imshow(error_maps[j], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[3,j].set_title(f'error ({maps[j]}, sh{sh_deg})')
        axes[3,j].axis('off')
    
    plt.tight_layout()
    plot_file = Path(dataset.model_path) / folder_plots / f"{view_idx:05d}_kappa{kappa:.2f}_SH{sh_deg}.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    exit(0)


#################################################
# GPU batch wise spherical harmonics transform  #
#   (modification from s2fft)                   #
#################################################
def unextend(f_ext: torch.Tensor, L: int, sampling: str = "mw") -> torch.Tensor:
    if sampling.lower() == "mw":
        return f_ext[:, 0:L, :]

    elif sampling.lower() == "mwss":
        return f_ext[:, 0 : L + 1, :]

    else:
        raise ValueError(
            "Only mw and mwss supported for periodic extension "
            f"(not sampling={sampling})"
        )


def upsample_by_two_mwss_ext(f_ext: torch.Tensor, L: int) -> torch.Tensor:
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = torch.fft.fftshift(torch.fft.fft(f_ext, axis=-2, norm="forward"), dim=[-2])

    ntheta_ext_up = 2 * ntheta_ext
    f_ext_up = torch.zeros(
        (f_ext.shape[0], ntheta_ext_up, nphi), dtype=torch.complex128, device=f_ext.device
    )
    f_ext_up[:, L : ntheta_ext + L, :nphi] = f_ext[:, 0:ntheta_ext, :nphi]
    return torch.conj(
        torch.fft.fft(
            torch.fft.ifftshift(torch.conj(f_ext_up), dim=[-2]),
            axis=-2,
            norm="backward",
        )
    )


def periodic_extension_spatial_mwss(
    f: torch.Tensor, L: int, spin: int = 0
) -> torch.Tensor:
    ntheta = L + 1
    nphi = 2 * L
    ntheta_ext = 2 * L

    f_ext = torch.zeros((f.shape[0], ntheta_ext, nphi), dtype=torch.complex128, device=f.device)
    f_ext[:, 0:ntheta, 0:nphi] = f[:, 0:ntheta, 0:nphi]
    f_ext[:, ntheta:, 0 : 2 * L] = (-1) ** spin * torch.fft.fftshift(
        torch.flip(f[:, 1 : ntheta - 1, 0 : 2 * L], dims=[-2]), dim=[-1]
    )
    return f_ext


def upsample_by_two_mwssCUDA(f: torch.Tensor, L: int, spin: int = 0) -> torch.Tensor:
    if f.ndim == 2:
        f = torch.unsqueeze(f, 0)
    f_ext = periodic_extension_spatial_mwss(f, L, spin)
    f_ext = upsample_by_two_mwss_ext(f_ext, L)
    f_ext = unextend(f_ext, 2 * L, sampling="mwss")
    return torch.squeeze(f_ext)


def forward_transform_torchCUDA(
	f: torch.Tensor,
    kernel: torch.Tensor,
    L: int,
) -> torch.Tensor:
    f = upsample_by_two_mwssCUDA(f, L)

    ftm = torch.fft.rfft(torch.real(f), axis=-1, norm="backward")
    ftm = ftm[...,:-1]

    if ftm.ndim == 2:
        ftm = torch.unsqueeze(ftm, 0)
        flm = torch.zeros((1,L**2), dtype=kernel.dtype, device=kernel.device)
    elif ftm.ndim == 3:
        flm = torch.zeros((ftm.shape[0],L**2), dtype=kernel.dtype, device=kernel.device)
    else:
        raise NotImplementedError
    
    ids_real = [l*L+m for l in range(L) for m in range(l+1)]
    ids_pos_m = [l*(l+1)+m for l in range(L) for m in range(l+1)]
    ids_imag = [l*(L-1)+m-1 for l in range(L) for m in range(1,l+1)]
    ids_neg_m = [l*(l+1)-m for l in range(L) for m in range(1,l+1)]
    
    flm[...,ids_pos_m] = torch.einsum(
        "...tlm, ...tm -> ...lm", kernel, ftm.real.to(dtype=kernel.dtype)
    ).reshape(ftm.shape[0],-1)[...,ids_real]
    flm[...,ids_neg_m] = -torch.einsum(
        "...tlm, ...tm -> ...lm", kernel[...,1:], ftm.imag[...,1:].to(dtype=kernel.dtype)
    ).reshape(ftm.shape[0],-1)[...,ids_imag]
    
    return flm
###############################################


def compute_gaussian_representations(gaussians, scene, dataset, pipe, sh_deg=0, kappa=8):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras()
    viewpoint_stack = train_cameras.copy()

    means3D = gaussians.get_xyz

    if sh_deg > 0:
        # set up von Mises Fisher
        vmf_norm_fac = np.exp(-kappa)
        vmf_pdf_batch = lambda mu,x: vmf_norm_fac * torch.exp(kappa*(torch.einsum("bdc,bd->bc", x, mu)))

        # set up s2fft
        sampling = "mwss"
        sqrt2=1.4142135623730951
        theta_array = thetas(sh_deg+1, sampling=sampling)
        phi_array = phis_equiang(sh_deg+1, sampling=sampling)
        theta, phi = np.meshgrid(theta_array, phi_array, indexing="ij")
        sample_points = torch.tensor(
            [np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)], dtype=means3D.dtype, device=means3D.device
        ).reshape(3,-1).expand(means3D.shape[0],-1,-1)
        forward_kernel = torch.tensor(spin_spherical_kernel(
            sh_deg+1, forward=True, reality=True, sampling=sampling
        ), dtype=means3D.dtype, device=means3D.device)
        forward_kernel[...,1:] *= sqrt2

        # the visibility signal on spheres around the Gaussians
        f_error = torch.zeros(size=(means3D.shape[0],)+theta.shape+(6,), dtype=means3D.dtype, device=means3D.device)
        f_visibility = torch.zeros(size=(means3D.shape[0],)+theta.shape+(6,), dtype=means3D.dtype, device=means3D.device)
    else:
        f_error = torch.zeros(size=(means3D.shape[0],1,1,6), dtype=means3D.dtype, device=means3D.device)
        f_visibility = torch.zeros(size=(means3D.shape[0],1,1,6), dtype=means3D.dtype, device=means3D.device)


    fov_counter = torch.zeros(means3D.shape[0], dtype=means3D.dtype, device=means3D.device)
    for i in tqdm(range(len(viewpoint_stack)), desc="Project training viewpoints"):
        viewpoint = viewpoint_stack[i]
        gt_rgb = viewpoint.original_image.to("cuda")

        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_rgb = render_pkg["render"].clamp(0.0, 1.0)
        error_rgb = torch.mean(torch.abs(pred_rgb - gt_rgb), axis=0)

        if sh_deg > 0:
            # 1. training view direction for each Gaussian
            dir_pp = (means3D - viewpoint.camera_center.expand(means3D.shape[0], -1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

        # 2. visibility for each training view for each Gaussian
        bp_error_pkg = backproject(viewpoint, gaussians, pipe, background, map_to_bp=error_rgb)
        bp_error_scalars = torch.stack([bp_error_pkg["bp_scalars"][:,VisMapMode.MAX],
                                        bp_error_pkg["bp_scalars"][:,VisMapMode.MAXnoAlpha],
                                        bp_error_pkg["bp_scalars"][:,VisMapMode.SUM],
                                        bp_error_pkg["bp_scalars"][:,VisMapMode.SUMnoAlpha],
                                        bp_error_pkg["bp_scalars"][:,VisMapMode.SUM] / bp_error_pkg["bp_scalars"][:,VisMapMode.PixCount],
                                        bp_error_pkg["bp_scalars"][:,VisMapMode.SUMnoAlpha] / bp_error_pkg["bp_scalars"][:,VisMapMode.PixCount]], dim=1).nan_to_num(nan=0.0)
        fov_counter += bp_error_pkg["visibility_filter"].to(dtype=int)
        bp_visibility_pkg = backproject(viewpoint, gaussians, pipe, background)
        bp_visibility_scalars = torch.stack([bp_visibility_pkg["bp_scalars"][:,VisMapMode.MAX],
                                             bp_visibility_pkg["bp_scalars"][:,VisMapMode.MAXnoAlpha],
                                             bp_visibility_pkg["bp_scalars"][:,VisMapMode.SUM],
                                             bp_visibility_pkg["bp_scalars"][:,VisMapMode.SUMnoAlpha],
                                             bp_visibility_pkg["bp_scalars"][:,VisMapMode.SUM] / bp_visibility_pkg["bp_scalars"][:,VisMapMode.PixCount],
                                             bp_visibility_pkg["bp_scalars"][:,VisMapMode.SUMnoAlpha]  / bp_visibility_pkg["bp_scalars"][:,VisMapMode.PixCount]], dim=1).nan_to_num(nan=0.0)
        
        # 3.1. accumulate the visibility information for the Gaussians
        if sh_deg > 0:
            pdf_vals = vmf_pdf_batch(dir_pp_normalized, sample_points).reshape((-1,)+theta.shape)
            pdf_vals = pdf_vals.unsqueeze(-1)
        else:
            # ablation of direction dipendence
            pdf_vals = 1.0
        f_error += bp_error_scalars.reshape(-1,1,1,6) * pdf_vals
        f_visibility = torch.maximum(f_visibility, bp_visibility_scalars.reshape(-1,1,1,6) * pdf_vals)
        torch.cuda.empty_cache()
    f_error /= len(viewpoint_stack)

    if sh_deg > 0:
        # 3.2. convert the visibility information (signal on sphere) to spherical harmonic coefficients
        # compute the spherical harmonic coefficients batch wise to avoid an explosion of memory use
        flm_error = torch.zeros((f_error.shape[0],(sh_deg+1)**2, 6), dtype=f_error.dtype, device=f_error.device)
        flm_visibility = torch.zeros((f_error.shape[0],(sh_deg+1)**2, 6), dtype=f_error.dtype, device=f_error.device)
        batch_size = 5000
        for i in range(0, f_error.shape[0], batch_size):
            for j in range(6):
                f_b = f_error[i:i+batch_size,...,j]
                flm_error[i:i+batch_size,...,j] = forward_transform_torchCUDA(f_b, forward_kernel, sh_deg+1)
                f_b = f_visibility[i:i+batch_size,...,j]
                flm_visibility[i:i+batch_size,...,j] = forward_transform_torchCUDA(f_b, forward_kernel, sh_deg+1)
    else:
        # direction independent
        flm_error = f_error.reshape(-1,6)
        flm_visibility = f_visibility.reshape(-1,6)

    gaussian_representations = {
        "fov_counter": fov_counter,
        "visibility": flm_visibility,
        "error": flm_error,
    }
    return gaussian_representations


def compute_and_store_gaussian_representations(dataset, pipe, sh_deg=0, kappa=8, store_splits=["test"], plot_splits=[]):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    gaussian_representations = compute_gaussian_representations(gaussians, scene, dataset=dataset, pipe=pipe, sh_deg=sh_deg, kappa=kappa)


    if "test" in store_splits:
        store_feature_maps(dataset, gaussians, pipe, gaussian_representations, scene.getTestCameras().copy(), sh_deg, kappa, depth_scale=scene.depth_scale,
                           folder_npz=f"PRIMU/feature_maps_kappa{kappa:.2f}_sh{sh_deg}/test",
                           plot="test" in plot_splits, folder_plots=f"PRIMU/plots/feature_maps_kappa{kappa:.2f}_sh{sh_deg}/test")
    if "train" in store_splits:
        store_feature_maps(dataset, gaussians, pipe, gaussian_representations, scene.getTrainCameras().copy(), sh_deg, kappa, depth_scale=scene.depth_scale,
                           folder_npz=f"PRIMU/feature_maps_kappa{kappa:.2f}_sh{sh_deg}/train",
                           plot="test" in plot_splits, folder_plots=f"PRIMU/plots/feature_maps_kappa{kappa:.2f}_sh{sh_deg}/train")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Create PRIMU uncertainty feature maps")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--sh_deg", type=int, default=0, help="spherical harmonics degree for direction dependent feature maps")
    parser.add_argument("--kappa", type=float, default=8, help="kappa for spread of von Mises Fisher distribution (for direction dependent feature maps)")
    parser.add_argument("--store", nargs="+", type=str, default=["test"], help="Split of views to store the feature maps ('test', 'train').")
    parser.add_argument("--plot", nargs="+", type=str, default=[], help="Split of views to plot the feature maps ('test', 'train').")
    args = parse_args_with_cfg_args(parser)
    
    print("Create PRIMU uncertainty feature maps " + args.model_path)

    with torch.no_grad():
        compute_and_store_gaussian_representations(lp.extract(args), pp.extract(args), sh_deg=args.sh_deg, kappa=args.kappa,
                                                   store_splits=args.store, plot_splits=args.plot)