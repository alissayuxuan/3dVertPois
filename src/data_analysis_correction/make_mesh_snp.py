import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import No_Logger, Log_Type, NII, BIDS_FILE, POI, Location, Vertebra_Instance, v_idx_order
import pyvista as pv
from utils.filepaths import search_path, search_path_single
from utils.poi_plotter import *
from TPTBox.core.vert_constants import COORDINATE, Location
from joblib import Parallel, delayed
from TPTBox import BIDS_FILE, POI, BIDS_Global_info, Log_Type, No_Logger
from TPTBox.mesh3D.mesh import Mesh3D, POIMesh, SegmentationMesh
from TPTBox.mesh3D.mesh_colors import get_color_by_label
from dataclasses import dataclass, field
from TypeSaveArgParse import Class_to_ArgParse
from tqdm import tqdm

logger = No_Logger(prefix="verse_cseg_meshsnp")


def _add_mesh(pl, mesh: pv.PolyData | SegmentationMesh, color: str, opacity: float = 1.0):
    if isinstance(mesh, Mesh3D):
        mesh = mesh.mesh
    pl.add_mesh(mesh, opacity=opacity, color=color)


def make_single_poi_vert_mesh(subreg_nii: NII, vert_nii: NII, v: int, poi: POI, html_out: Path):
    pv.start_xvfb()  # type: ignore
    pl: pv.Plotter = pv.Plotter()  # type: ignore
    pl.set_background("black", top=None)
    pl.add_axes()

    inst_sem = vert_nii.extract_label(v) * subreg_nii

    meshes = {}
    for s in range(41, 50):
        try:
            mesh = SegmentationMesh.from_segmentation_nii(inst_sem.extract_label(s), rescale_to_iso=False)
            meshes[s] = mesh
            _add_mesh(pl, mesh, color=get_color_by_label(s).rgb, opacity=1.0)
        except ValueError:
            logger.print(f"Subject {v}: No mesh for subregion {s}, skipping", Log_Type.WARNING)
    poiv = poi.extract_region(v)
    print(poiv.keys_subregion())
    for idx, s in enumerate(poiv.keys_subregion()):
        poimesh = POIMesh(
            poiv.extract_subregion(s),
            rescale_to_iso=False,
            regions=None,
            subregions=None,
            size_factor=2.5,
        )
        _add_mesh(pl, poimesh, color=get_color_by_label(idx).rgb, opacity=0.99)
    pl.export_html(str(html_out))
    logger.print(f"Saved scene into {html_out}", Log_Type.SAVE)


def make_poi_vert_mesh(poi_ref: POI, vert_nii: NII, subreg_nii: NII, subject_id: str, out_snp_labellambda: callable):
    labels = vert_nii.unique()
    for v in tqdm(labels, desc=f"Making mesh for {subject_id}"):
        if v <= 7:
            continue
        if v >= 29 or v in [26, 27]:
            continue
        html_out = out_snp_labellambda(v)
        if html_out.exists():
            logger.print(f"{subject_id} - vert {v}: POI mesh html already exists, skipping: {html_out}", Log_Type.WARNING)
            continue
        make_single_poi_vert_mesh(subreg_nii, vert_nii, v, poi_ref, html_out)


def _proc(subject: Path, ds_dir: Path, opt):
    subject_id = subject.name
    if not subject.is_dir():
        return

    # find all rawdata ct files
    img_paths = search_path(ds_dir / opt.rawdata / subject_id, f"{subject_id}*_ct.nii.gz")
    assert len(img_paths) > 0, f"No CT image found for subject {subject_id}"
    for img_path in img_paths:
        subject_ct_id = img_path.name.split(".")[0]
        #
        out_snp = ds_dir.joinpath(opt.out_parent, opt.der_out_prefix + opt.der_poi_mainpred, subject_ct_id, f"{subject_ct_id}_snp.html")

        def out_snp_labellambda(x):
            return ds_dir.joinpath(
                opt.out_parent, opt.der_out_prefix + opt.der_poi_mainpred, subject_ct_id, f"{subject_ct_id}_label-{x}_snp.html"
            )

        if out_snp.exists():
            logger.print(f"{subject_ct_id}: POI mesh html already exists, skipping: {out_snp}", Log_Type.WARNING)
            continue
        out_snp.parent.mkdir(parents=True, exist_ok=True)
        #

        # logger.print("Subject:", subject_ct_id)
        # img_path = search_path_single(ds_dir / RAWDATA / subject_id, f"{subject_id}*_ct.nii.gz")
        img_bidsf = BIDS_FILE(img_path, dataset=ds_dir)
        split_info = img_bidsf.info["split"] if "split" in img_bidsf.info else None
        vert_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "vert"}, parent=opt.der_vert)
        # VERT MASK IS NOT MODIFIED, so just copy
        assert vert_path.exists(), f"{subject_ct_id}: Original vert mask does not exist: {vert_path}"
        subreg_path = img_bidsf.get_changed_path(file_type="nii.gz", bids_format="msk", info={"seg": "subreg"}, parent=opt.der_subreg)
        assert subreg_path.exists(), f"{subject_ct_id}: NO subreg mask exists: {subreg_path}"

        # load both masks
        subreg_nii = NII.load(subreg_path, seg=True).reorient_()
        subreg_nii.map_labels_({51: 49, 50: 49}, verbose=False)
        vert_nii = NII.load(vert_path, seg=True).reorient_()
        subreg_nii.assert_affine(other=vert_nii, verbose=logger, raise_error=True)

        poi_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_mainpred
        )
        if not poi_ref_path.exists():
            poi_ref_path = img_bidsf.get_changed_path(
                file_type="json",
                bids_format="poi",
                info={"seg": "vert", "mod": None, "source": "deterministic"},
                parent=opt.der_poi_mainpred,
            )
        poi_proj_ref_path = img_bidsf.get_changed_path(
            file_type="json", bids_format="poi", info={"seg": "vert", "mod": "ct"}, parent=opt.der_poi_mainpred + "_sproj"
        )
        assert poi_ref_path.exists(), f"{subject_ct_id}: Main POI prediction file does not exist: {poi_ref_path}"
        if poi_proj_ref_path.exists():
            poi_ref = POI.load(poi_proj_ref_path).reorient_()
        else:
            poi_ref = POI.load(poi_ref_path).reorient_()

        make_poi_vert_mesh(poi_ref, vert_nii, subreg_nii, subject_ct_id, out_snp_labellambda)


@dataclass
class InferenceConfig(Class_to_ArgParse):
    root: Path = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/")
    rawdata: str = "rawdata"
    der_subreg: str = "derivatives_combined"  # "derivatives_subreg"
    der_vert: str = "derivatives_combined"
    #######
    der_poi_mainpred: str = "derivatives_poi_deterministic"
    # "derivatives_poi_surface_cc3-v0_flipped"
    # "derivatives_poi_surface_cc3-bs32-v1_flipped"
    # "derivatives_poi_surface_neighbor_cc3-v1_flipped"
    # ]
    #######
    # "derivatives_poi_automatic_correction"
    # "derivatives_poi_surface-neighbor-neighaug-project_gt_cc3-exclude6-v2"
    der_out_prefix: str = "TEST_"

    out_parent: str = "snaps_mesh-poi"

    ignore_poi: list[Location] = field(
        default_factory=lambda: [
            Location.Vertebra_Direction_Inferior,
            Location.Vertebra_Direction_Posterior,
            Location.Vertebra_Direction_Right,
            Location.Vertebra_Corpus,
        ]
    )
    datasets: list[str] = field(
        default_factory=lambda: [
            "dataset-verse19training_1mmiso",
            # "dataset-verse20training_1mmiso",
            # "dataset-verse19validation_1mmiso",
            # "dataset-verse20validation_1mmiso",
            # "dataset-verse19test_1mmiso",
            # "dataset-verse20test_1mmiso",
        ]
    )

    num_threads: int = 4
    cprofile_this: bool = False


if __name__ == "__main__":
    import cProfile

    opt = InferenceConfig.get_opt()
    if opt.cprofile_this:
        opt.num_threads = 1  # avoid multithreading issues with cprofile

    for ds_dir in opt.root.iterdir():
        if not ds_dir.is_dir():
            continue
        if not ds_dir.name.startswith("dataset-"):
            continue

        if ds_dir.name not in opt.datasets:
            continue

        # has rawdata folder
        raw_dir = ds_dir.joinpath(opt.rawdata)
        assert raw_dir.exists(), f"No rawdata dir in dataset dir: {ds_dir}"

        logger.print("Processing dataset dir:", ds_dir.name, Log_Type.STAGE)

        subjects = list(raw_dir.iterdir())

        if not opt.cprofile_this:
            Parallel(n_jobs=opt.num_threads)(delayed(_proc)(subject, ds_dir, opt) for subject in subjects)
        else:
            for subject in subjects:
                if "e014" not in subject.name:
                    continue
                with cProfile.Profile() as pr:
                    _proc(subject, ds_dir, opt)
                pr.dump_stats(
                    f"/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/cprofiles/{ds_dir.name}_{subject.name}_poi_report.prof"
                )
                break

        # break
