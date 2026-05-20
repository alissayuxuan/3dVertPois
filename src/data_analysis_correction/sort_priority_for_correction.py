import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
import pandas as pd
from dataclasses import dataclass
from TPTBox import Location, NII, POI, Vertebra_Instance, No_Logger, Log_Type
from TPTBox.core.vert_constants import DIRECTIONS, COORDINATE
from pathlib import Path
import numpy as np
from report_utils import convert_agg_report_to_reported_bool_dict, is_poi_reported, load_agg_report_df
from utils.filepaths import search_path

logger = No_Logger(prefix="logic_report")


if __name__ == "__main__":
    ROOT = Path("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/")
    PREFIX = "TEST_"

    name = "aggregated_poi_report.xlsx"
    ps = search_path(ROOT, query=f"**/{name}")
    # der_names = ROOT.glob(f"{PREFIX}*")

    # for der in der_names:
    #    if not der.is_dir():
    #        continue
    #    if "derivatives" not in der.name:
    #        continue
    #    if "poi" not in der.name:
    #        continue

    #    der = der.name.split(f"{PREFIX}")[1]
    #    logger.print(f"{der}", Log_Type.STAGE)
    for p in ps:
        df = pd.read_excel(p)
        # df = load_agg_report_df(str(ROOT), PREFIX, der)
        if df is not None:
            report_der2dict_sum_per_vert = {}
            for row in df.iterrows():
                subject_name = row[1]["subject_name"]
                vert = row[1]["vertebra"]
                location = Location(row[1]["location"])
                severity = row[1]["severity"]
                if subject_name not in report_der2dict_sum_per_vert:
                    report_der2dict_sum_per_vert[subject_name] = {}
                if vert not in report_der2dict_sum_per_vert[subject_name]:
                    report_der2dict_sum_per_vert[subject_name][vert] = 0
                if severity > 0.0:
                    report_der2dict_sum_per_vert[subject_name][vert] += 1

            # export report_der2dict_sum_per_vert as df again
            report_der2dict_sum_per_vert_df = pd.DataFrame(
                [
                    {"subject_name": subject_name, "vertebra": vert, "num_reported_errors": num_errors}
                    for subject_name, vert_dict in report_der2dict_sum_per_vert.items()
                    for vert, num_errors in vert_dict.items()
                ]
            )
            out_excel = Path(p).parent.joinpath("aggregated_report_num_errors_per_vert.xlsx")
            # sort report_der2dict_sum_per_vert_df by descending num_reported_errors
            report_der2dict_sum_per_vert_df = report_der2dict_sum_per_vert_df.sort_values(by="num_reported_errors", ascending=False)
            report_der2dict_sum_per_vert_df.to_excel(out_excel, index=False)
        # break
