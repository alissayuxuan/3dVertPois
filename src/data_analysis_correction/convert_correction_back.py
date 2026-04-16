import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import POI, POI_Global
from utils.filepaths import search_path, search_path_single


def convert_correction_back(target_dir: Path, reference_poi_f: Path):
    corrected_files = search_path(str(target_dir), "Point*_.mrk.json")
    if len(corrected_files) == 0:
        print(f"No corrected POI files found in {target_dir}")
        return None

    assert reference_poi_f.exists(), f"Reference POI file not found: {reference_poi_f}"

    try:
        reference_poi = POI.load(reference_poi_f)
    except Exception as e:
        print(f"Error loading reference POI: {str(e)}")
        return

    poi2 = None
    for corrected_file in corrected_files:
        print(f"Processing {corrected_file}")
        try:
            poi = POI_Global.load(corrected_file)
        except Exception as e:
            print(f"Error loading corrected POI: {str(e)}")
            continue

        poiadd = poi.to_local(reference_poi)
        if poi2 is None:
            poi2 = poiadd
        else:
            poi2.centroids.update(poiadd.centroids)
        # aggregate into a single poi object
    return poi2


ROOT = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
RAWDATA = "rawdata"
DERIV_REF = "derivatives_poi_automatic_correction-v3-6-onlygood"

# correction specific
DS_NAME = "dataset-verse19training_1mmiso"
SOURCE_ROOT = f"/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/{DS_NAME}/TANJA/correction_VERSE-pois/"
TARGET_DER = "derivatives_manual_tanja"


def single_test():
    target_dir = Path(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/dataset-verse19training_1mmiso/TANJA/correction_VERSE-pois/sub-verse097_ct/"
    )
    reference_poi_f = Path(
        "/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/dataset-verse19training_1mmiso/derivatives_poi_automatic_correction-v3-6-onlygood/sub-verse097/sub-verse097_mod-ct_seg-vert_poi.json"
    )

    poi = convert_correction_back(target_dir, reference_poi_f)

    if poi is not None:
        out_path = Path(
            "/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/TEST/sub-verse097_mod-ct_seg-vert_poi.json"
        )
        poi.save(out_path)


if __name__ == "__main__":
    # single_test()

    # find all dirs in SOURCE_ROOT
    source_dirs = [d for d in Path(SOURCE_ROOT).iterdir() if d.is_dir()]
    for sdir in source_dirs:
        subject = sdir.name.split("_")[0]
        subject_extended = sdir.name.split("_ct")[0]
        print(f"Processing {subject}")

        target_dir = sdir
        reference_poi_f = search_path_single(
            Path(ROOT).joinpath(DS_NAME, DERIV_REF, subject),
            f"{subject_extended}*_mod-ct*_seg-vert*_poi.json",
            raise_missing=True,
            verbose=True,
        )
        assert reference_poi_f is not None, f"Reference POI file not found for {subject} in {DERIV_REF}"
        out_path = Path(ROOT).joinpath(DS_NAME, TARGET_DER, subject, reference_poi_f.name)
        if out_path.exists():
            print(f"Output file {out_path} already exists, skipping {subject}")
            continue

        poi = convert_correction_back(target_dir, reference_poi_f)

        if poi is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            poi.save(out_path)
