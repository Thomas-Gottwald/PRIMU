import itertools
import json
import numpy as np
import pickle
import re
import scipy as sp
from argparse import ArgumentParser
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from utils.ause_torch import ause_torch


def evaluate_ue_metrics(y, y_pred):
    # Pearson correlation
    pearson = sp.stats.pearsonr(y_pred, y)

    # Spearmen rank correlation
    spearman = sp.stats.spearmanr(y_pred, y)

    # AUSE (area under specification error)
    ause, _, _ = ause_torch(y, y_pred)

    eval_dict = {
        "pearson": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "AUSE": float(ause),
    }

    return eval_dict


def get_indices_from_stack(i_stack):
    if "+" in i_stack:
        i_list = [int(i_s) for i_s in i_stack.split("+")]
    elif "-" in i_stack:
        if not re.match("^[0-9]*-[0-9]*$", i_stack):
            print(f"    WARNING: Invalid holdout index stacking encountered: {i_stack}! Combining with minus('-') only supports two given integers!")
        i_stack_list = i_stack.split("-")
        i_list = [i_r for i_r in range(int(i_stack_list[0]), int(i_stack_list[1])+1)]
    else:
        i_list = [int(i_stack)]

    return i_list


def load_feature_maps(maps_files, X_map_names, error_type="rgb"):
    y_error_list = []
    y_shapes_list = []
    X_maps_list = []
    for f in maps_files:
        npz_data = np.load(f)

        if error_type.lower() == "rgb":
            gt = npz_data["gt_rgb"]
            pred = npz_data["pred_rgb"]
            y_error = np.abs(pred-gt).mean(0)
        elif error_type.lower() == "depth":
            gt = npz_data["gt_depth"]
            pred = npz_data["pred_depth"]
            y_error = np.abs(pred-gt)
        else:
            raise ValueError("Only 'rgb' and 'depth' error_type is supported!")
        y_error_vec = y_error.reshape(-1)
        
        X_maps = np.zeros(shape=y_error.shape+(len(X_map_names),))
        for j,km in enumerate(X_map_names):
            X_maps[:,:,j] = npz_data[km]
        X_maps_mat = X_maps.reshape(-1, X_maps.shape[-1])

        y_error_list.append(y_error_vec)
        y_shapes_list.append(y_error.shape)
        X_maps_list.append(X_maps_mat)
    y_error = np.concat(y_error_list)
    X_maps = np.concat(X_maps_list, axis=0)

    return y_error, X_maps, y_shapes_list


def train_regressor(y_train, X_train, regressor_model="hgbr"):
    if regressor_model == "hgbr":
        common_params = {
            "max_iter": 150,
            "learning_rate": 0.3,
            "validation_fraction": 0.2,
            "random_state": 42,
            "categorical_features": None,
            "scoring": "neg_root_mean_squared_error",
        }
        model = HistGradientBoostingRegressor(early_stopping=False, **common_params)
    elif regressor_model == "lin":
        model = LinearRegression()
    else:
        raise NotImplementedError()
    
    model.fit(X_train, y_train)

    return model


def evaluate_regressor(model, maps_files, X_map_names, error_type="rgb", save_maps_path=None):
    eval_dict = dict()
    eval_views_dict = dict()
    # for all holdout views
    for f in maps_files:
        scene = f.parents[3].name
        view_idx =  int(f.stem[:5])
        if scene not in eval_views_dict:
            eval_views_dict[scene] = dict()

        # load uncertainty feature maps and compute error
        y_error, X_maps, y_shapes = load_feature_maps([f], X_map_names, error_type)

        # regression model prediction on holdout view
        y_pred = model.predict(X_maps)
        
        # evaluate UE metrics
        v_eval_dict = evaluate_ue_metrics(y_error, y_pred)
        for k in v_eval_dict:
            if k in eval_dict:
                eval_dict[k] += v_eval_dict[k]
            else:
                eval_dict[k] = v_eval_dict[k]
        eval_views_dict[scene][view_idx] = v_eval_dict

        if save_maps_path is not None:
            # save UE maps
            y_pred_map = y_pred.reshape(y_shapes[0])
            save_maps_path.mkdir(parents=True, exist_ok=True)
            np.save(save_maps_path / f"{f.stem[:5]}_ue_map.npy", y_pred_map)
    for k in eval_dict:
        eval_dict[k] /= len(maps_files)
    eval_dict["test_holdout_views"] = eval_views_dict

    return eval_dict


def save_eval_dict(eval_dict, regressor_info_dict, save_eval_path, split, idx_stack):
    eval_dict["regressor_info"] = regressor_info_dict

    save_file = save_eval_path / split / f"{idx_stack}_eval.json"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text(json.dumps(eval_dict, indent=2))
    print(f"Saved UE metrics to: {save_file}")


def save_regressor_model(model, regressor_info_dict, save_regressor_file: Path):
    store = {
        "model": model,
        "regressor_info": regressor_info_dict,
    }
    save_regressor_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(store, open(save_regressor_file, "wb"))


def load_regressor_model(load_regressor_file):
    load = pickle.load(open(load_regressor_file, "rb"))
    model = load["model"]
    regressor_info_dict = load["regressor_info"]

    return model, regressor_info_dict


def train_and_evaluate_regressor_models(error_type, source_paths, holdout_indices, sh_deg, kappa, x_maps, map_variants, regressor_model,
                                        experiment_tag: str="", save_ue_maps=False, save_regressor=False, eval_on_train=False):
    X_map_names = []
    for x_map in x_maps:
        if "*" in x_map:
            X_map_names.extend([x_map.replace("*", app) for app in map_variants])
        else:
            X_map_names.append(x_map)

    for idx_stack in holdout_indices:
        idx_list = get_indices_from_stack(idx_stack)
        load_iterate = list(itertools.product(*[source_paths, idx_list]))

        # Train holdout view set:
        # collect uncertainty feature map files for regressor training set
        train_maps_files = []
        for p,i in load_iterate:
            maps_path = Path(p) / "PRIMU" / f"feature_maps_kappa{kappa:.2f}_sh{sh_deg}" / "test"
            maps_file = maps_path / f"{i:05d}_kappa{kappa:.2f}_SH{sh_deg}.npz"
            train_maps_files.append(maps_file)
        
        # load uncertainty feature maps and compute error (target for regression)
        y_error_train, X_maps_train, _ = load_feature_maps(train_maps_files, X_map_names, error_type)

        # train regression model
        model = train_regressor(y_error_train, X_maps_train, regressor_model)

        # create dictionary with regressor training info
        train_views_dict = dict()
        for f in train_maps_files:
            scene = f.parents[3].name
            view_idx =  int(f.stem[:5])
            if scene not in train_views_dict:
                train_views_dict[scene] = []
            train_views_dict[scene].append(view_idx)
        regressor_info_dict = {
            "regressor_model": regressor_model,
            "error_type": error_type,
            "sh_deg": sh_deg,
            "kappa": kappa,
            "X_map_names": X_map_names,
            "training_holdout_views": train_views_dict,
        }

        # setup output paths
        base_path = Path(source_paths[0]).parent / "_".join(train_views_dict.keys()) / "PRIMU"
        experiment_dir = f"{error_type}_{regressor_model}_kappa{kappa:.2f}_sh{sh_deg}"
        if experiment_tag != "":
            experiment_dir = Path(experiment_tag) / experiment_dir
        save_eval_path = base_path / "eval" / experiment_dir
        if save_ue_maps:
            save_maps_path = base_path / "ue_maps" / experiment_dir / f"{idx_stack}_training_holdout_views"
        else:
            save_maps_path = None
        if save_regressor:
            save_regressor_file = base_path / "regressor" / experiment_dir / f"{idx_stack}_regressor.p"
            save_regressor_model(model, regressor_info_dict, save_regressor_file)

        if eval_on_train:
            # evaluate regression model on training data
            eval_train_dict = evaluate_regressor(model, train_maps_files, X_map_names, error_type, save_maps_path)
            save_eval_dict(eval_train_dict, regressor_info_dict, save_eval_path, "train", idx_stack)

        # Test holdout view set: 
        # collect uncertainty feature map files for regressor testing set
        test_maps_files = []
        for p in source_paths:
            maps_path = Path(p) / "PRIMU" / f"feature_maps_kappa{kappa:.2f}_sh{sh_deg}" / "test"
            maps_files = [f for f in maps_path.glob("*.npz") if f not in train_maps_files]
            test_maps_files.extend(maps_files)
        test_maps_files = sorted(test_maps_files)

        # evaluate regression model on test data
        eval_test_dict = evaluate_regressor(model, test_maps_files, X_map_names, error_type, save_maps_path)
        save_eval_dict(eval_test_dict, regressor_info_dict, save_eval_path, "test", idx_stack)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train regression model on uncertainty feature maps to predict error")
    parser.add_argument("-e", "--error_type", type=str, default="rgb", help="The type of error that should be predicted ('rgb', 'depth').")
    parser.add_argument("-s", "--scene_paths", nargs="+", type=str,
                        help="Path to GS scene output directories." \
                        "When multiple scene paths are given the regressor is trained on views from all the scenes.")
    parser.add_argument("-i", "--holdout_indices", nargs="+", type=str, default=['0', '1', '2', '3'],
                        help="Indices of holdout views to train the regressor. Remaining holdout views are used for evaluation." \
                        "For each entry a separate regression model is trained." \
                        "To train regressors on multiple views use the syntax: '0-2' for a range and '0+1+3' for a selection.")
    parser.add_argument("--sh_deg", type=int, default=0, help="spherical harmonics degree for direction dependent feature maps")
    parser.add_argument("--kappa", type=float, default=8, help="kappa for spread of von Mises Fisher distribution (for direction dependent feature maps)")
    parser.add_argument("-x", "--x_maps", nargs="+", type=str,
                        default=["visibility_map_*_sh0", "visibility_counter", "error_bp_map_*_sh0"])
    parser.add_argument("-v", "--map_variants", nargs="+", type=str,
                        default=["MAX", "MAXnoAlpha", "SUM", "SUMnoAlpha", "MEAN", "MEANnoAlpha"])
    parser.add_argument("-r", "--regressor_model", type=str, default="hgbr",
                        help="Sklearn regression model used: hgbr=HistGradientBoostingRegressor, lin=LinearRegressor")
    parser.add_argument("--experiment_tag", type=str, default="", help="Optional tag to organize different experiments. Is added to the output path.")
    parser.add_argument("--save_ue_maps", action="store_true", help="Store uncertainty maps as npy.")
    parser.add_argument("--save_regressor", action="store_true", help="Save the regression model in a pickle file.")
    parser.add_argument("--eval_on_train", action="store_true", help="Evaluate also on the training holdout views. Metrics are stored in a separate json file.")
    args = parser.parse_args()

    train_and_evaluate_regressor_models(
        error_type=args.error_type,
        source_paths=args.source_paths,
        holdout_indices=args.holdout_indices,
        sh_deg=args.sh_deg,
        kappa=args.kappa,
        x_maps=args.x_maps,
        map_variants=args.map_variants,
        regressor_model=args.regressor_model,
        experiment_tag=args.experiment_tag,
        save_ue_maps=args.save_ue_maps,
        save_regressor=args.save_regressor,
        eval_on_train=args.eval_on_train
    )