import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import pandas as pd
from pathlib import Path
from utils.filepaths import search_path
from TPTBox import No_Logger, Log_Type
from tqdm import tqdm

IGNORE = ["cprofiles", "debug"]
logger = No_Logger(prefix="aggregate_poi_reports")


def aggregate_in_dir(dir: Path, ds_names: list[str]) -> None | pd.DataFrame:
    agg_df = None
    for ds_name in ds_names:
        basepath = dir.joinpath(ds_name)
        if not basepath.exists():
            continue
        df_ps = search_path(basepath, query="*poi_neighbor_angle_report.xlsx")
        for df_p in tqdm(df_ps, desc=f"Processing {ds_name}"):
            df = pd.read_excel(df_p)
            df["ds"] = ds_name
            if agg_df is None:
                agg_df = df
            else:
                agg_df = pd.concat([agg_df, df], ignore_index=True)
    return agg_df


if __name__ == "__main__":
    ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/")

    ds_names = [
        "dataset-verse19training_1mmiso",
        #"dataset-verse20training_1mmiso",
        #"dataset-verse19validation_1mmiso",
        #"dataset-verse20validation_1mmiso",
        #"dataset-verse19test_1mmiso",
        #"dataset-verse20test_1mmiso",
    ]

    for dir in ROOT.iterdir():
        if not dir.is_dir():
            continue

        # if "derivatives" not in dir.name:
        #    continue
        #
        # if "poi" not in dir.name:
        #    continue

        out_agg_df = dir.joinpath("aggregated_poi_neighbor_angle_report.xlsx")
        if out_agg_df.exists():
            continue

        logger.print(f"Processing directory: {dir}", Log_Type.STAGE)
        agg_df = aggregate_in_dir(dir, ds_names)
        if agg_df is not None:
            agg_df.to_excel(out_agg_df, index=False)
            logger.print(f"Aggregated report saved to {out_agg_df}", Log_Type.SAVE)
    # combined_report = pd.concat([report1, report2], ignore_index=True)
    # combined_report.to_excel("combined_logic_report.xlsx", index=False)
