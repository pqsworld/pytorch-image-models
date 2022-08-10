import numpy as np
import pandas as pd


results = {
    "results-imagenet.csv": [
        "results-imagenet-real.csv",
        "results-imagenetv2-matched-frequency.csv",
        "results-sketch.csv",
    ],
    "results-imagenet-a-clean.csv": [
        "results-imagenet-a.csv",
    ],
    "results-imagenet-r-clean.csv": [
        "results-imagenet-r.csv",
    ],
}


def diff(base_df, test_csv):
    base_models = base_df["model"].values
    test_df = pd.read_csv(test_csv)
    test_models = test_df["model"].values

    rank_diff = np.zeros_like(test_models, dtype="object")
    all_diff = np.zeros_like(test_models, dtype="object")
    pos_diff = np.zeros_like(test_models, dtype="object")
    neg_diff = np.zeros_like(test_models, dtype="object")

    for rank, model in enumerate(test_models):
        if model in base_models:
            base_rank = int(np.where(base_models == model)[0])
            all_d = test_df["all"][rank] - base_df["all"][base_rank]
            pos_d = test_df["pos"][rank] - base_df["pos"][base_rank]
            neg_d = test_df["neg"][rank] - base_df["neg"][base_rank]

            # rank_diff
            if rank == base_rank:
                rank_diff[rank] = f"0"
            elif rank > base_rank:
                rank_diff[rank] = f"-{rank - base_rank}"
            else:
                rank_diff[rank] = f"+{base_rank - rank}"

            # top1_diff
            if all_d >= 0.0:
                all_diff[rank] = f"+{all_d:.3f}"
            else:
                all_diff[rank] = f"-{abs(all_d):.3f}"

            if pos_d >= 0.0:
                pos_diff[rank] = f"+{pos_d:.3f}"
            else:
                pos_diff[rank] = f"-{abs(pos_d):.3f}"

            if neg_d >= 0.0:
                neg_diff[rank] = f"+{neg_d:.3f}"
            else:
                neg_diff[rank] = f"-{abs(neg_d):.3f}"

        else:
            rank_diff[rank] = ""
            all_diff[rank] = ""
            pos_diff[rank] = ""
            neg_diff[rank] = ""

    test_df["all_diff"] = all_diff
    test_df["pos_diff"] = pos_diff
    test_df["neg_diff"] = neg_diff
    test_df["rank_diff"] = rank_diff

    test_df["param_count"] = test_df["param_count"].map("{:,.2f}".format)
    test_df.sort_values("all", ascending=False, inplace=True)
    test_df.to_csv(test_csv, index=False, float_format="%.3f")


for base_results, test_results in results.items():
    base_df = pd.read_csv(base_results)
    base_df.sort_values("all", ascending=False, inplace=True)
    for test_csv in test_results:
        diff(base_df, test_csv)
    base_df["param_count"] = base_df["param_count"].map("{:,.2f}".format)
    base_df.to_csv(base_results, index=False, float_format="%.3f")
