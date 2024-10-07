from AU_recognizer import CONFIG, MODEL_FOLDER, P_PATH, F_OUTPUT, MF_MODEL, MF_FIT_MODE, MF_SAVE_IMAGES, MF_SAVE_MESH, \
    MF_SAVE_CODES, logger
from AU_recognizer.core.util import nect_config
from emoca.gdl_apps.EMOCA.utils.load import load_model
from emoca.gdl.datasets.ImageTestDataset import TestData
from pathlib import Path
from tqdm import auto
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test


def emoca_fit(fit_data, images_to_fit, project_data):
    logger.info(f"emoca fit of {images_to_fit}, with {fit_data}, and {project_data}")
    # data     return {
    #             MF_MODEL: self.model_combobox.get_value(),
    #             MF_SAVE_IMAGES: self.save_images.get_value(),
    #             MF_SAVE_CODES: self.save_codes.get_value(),
    #             MF_SAVE_MESH: self.save_mesh.get_value(),
    #             MF_FIT_MODE: self.fit_mode.get_value()
    #         }
    images_to_fit=str(images_to_fit.resolve())
    _project_name = str(project_data.sections()[0])
    project_path = Path(project_data[_project_name][P_PATH])
    path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER])
    model_name = fit_data[MF_MODEL]
    output_folder = project_path / F_OUTPUT
    mode = fit_data[MF_FIT_MODE]

    # 1) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    dataset = TestData(images_to_fit, face_detector="fan", max_detection=20)

    # 4) Run the model on the data
    for i in auto.tqdm(range(len(dataset))):
        try:
            batch = dataset[i]
        except:
            logger.warning(f"error during fitting {dataset.imagepath_list[i]}")
            continue
        vals, visdict = test(emoca, batch)
        current_bs = batch["image"].shape[0]

        for j in range(current_bs):
            name = batch["image_name"][j]

            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            if fit_data[MF_SAVE_MESH]:
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
            if fit_data[MF_SAVE_IMAGES]:
                save_images(output_folder, name, visdict, with_detection=True, i=j)
            if fit_data[MF_SAVE_CODES]:
                save_codes(Path(output_folder), name, vals, i=j)
    logger.warning(f"fitting done")
