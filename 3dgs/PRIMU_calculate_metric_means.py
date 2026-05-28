import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


def calculate_means(eval_dir: Path, metrics):
    eval_files = sorted(eval_dir.glob("*.json"))

    metric_values = {m: [] for m in metrics}
    for eval_file in eval_files:
        data = json.load(eval_file.open())
        for m in metrics:
            metric_values[m].append(data[m])

    mean_values = dict()
    for m in metrics:
        mean_values[m] = np.mean(metric_values[m])
        mean_values[m+"_std"] = np.std(metric_values[m], ddof=1)

    mean_file = eval_dir / "mean_eval.json"
    mean_file.write_text(json.dumps(mean_values, indent=2))
    print(f"Metric means saved to: {mean_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate the mean UE metrics from all json files the given directory")
    parser.add_argument("-d", "--dir", type=str, required=True, help="Directory containing the json files")
    parser.add_argument("-m", "--metrics", type=str, nargs="+", default=["pearson", "spearman", "AUSE"], help="List of UE metrics to calculate the mean of.")
    args = parser.parse_args()

    calculate_means(
        eval_dir=Path(args.dir),
        metrics=args.metrics
    )