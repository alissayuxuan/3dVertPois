from TPTBox import POI, POI_Global

if __name__ == "__main__":
    p = "/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/dataset-verse19training_1mmiso/TANJA/correction_VERSE-pois/sub-verse082_ct/"

    poi = POI_Global.load(p + "Point8_.mrk.json")

    ref_p = "/DATA/NAS/datasets_processed/CT_spine/dataset-verse-challenge/dataset-verse19training_1mmiso/derivatives_poi_automatic_correction-v3-6-onlygood/sub-verse082/"
    ref_poi = POI.load(ref_p + "sub-verse082_mod-ct_seg-vert_poi.json")

    poi2 = poi.to_local(ref_poi)

    print(poi2)

    poi2.save("/DATA/NAS/ongoing_projects/hendrik/poi_prediction/3dVertPois/data_analysis/TEST/sub-verse082_mod-ct_seg-vert_poi.json")
