from pathlib import Path

import torch
from tqdm import auto

from .datasets.ImageTestDataset import TestData
from .utils.model import save_obj, save_images, save_codes
from .utils.utility import load_model
from AU_recognizer.core.util import (logger, P_PATH, nect_config, CONFIG, MODEL_FOLDER, MF_MODEL, F_OUTPUT, MF_FIT_MODE,
                                     MF_SAVE_MESH, MF_SAVE_IMAGES, MF_SAVE_CODES)


def test_muscle(deca, img, muscle_images):
    img["image"] = img["image"].cuda()
    if len(img["image"].shape) == 3:
        img["image"] = img["image"].view(1, 3, 224, 224)
    vals = deca.encode(img, training=False)
    vals, visdict = decode_muscle(deca, vals, muscle_images, training=False)
    return vals, visdict


def decode_muscle(emoca, values, muscle_images, training=False):
    with torch.no_grad():
        values = emoca.decode(values, training=training)
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        visualizations, grid_image = emoca.visualization_checkpoint(
            values['verts'],
            values['trans_verts'],
            values['ops'],
            uv_detail_normals,
            values,
            0,
            "",
            "",
            save=False
        )

    return values, visualizations


def muscle_fit(fit_data, images_to_fit, project_data, muscle_images):
    logger.info(f"emoca fit of {images_to_fit}, with {fit_data}, and {project_data}")
    # data     return {
    #             MF_MODEL: self.model_combobox.get_value(),
    #             MF_SAVE_IMAGES: self.save_images.get_value(),
    #             MF_SAVE_CODES: self.save_codes.get_value(),
    #             MF_SAVE_MESH: self.save_mesh.get_value(),
    #             MF_FIT_MODE: self.fit_mode.get_value()
    #         }
    if isinstance(images_to_fit, list):
        # If it's a list, ensure each image path is resolved
        images_to_fit = [str(image.resolve()) for image in images_to_fit]
    else:
        # If it's a single path (folder), resolve it directly
        images_to_fit = str(images_to_fit.resolve())
    _project_name = str(project_data.sections()[0])
    project_path = Path(project_data[_project_name][P_PATH])
    path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER])
    model_name = fit_data[MF_MODEL]
    output_folder = project_path / F_OUTPUT / model_name
    mode = fit_data[MF_FIT_MODE]

    # 0) clear memory
    # Clear previous model if it exists
    torch.cuda.empty_cache()  # Clear any cached GPU memory

    # 1) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    dataset = TestData(images_to_fit, face_detector="fan", max_detection=20)

    # 3) Run the model on the data
    for i in auto.tqdm(range(len(dataset))):
        try:
            batch = dataset[i]
        except Exception as e:
            logger.warning(f"error during fitting {dataset.imagepath_list[i]}: {e}")
            continue
        vals, visdict = test_muscle(emoca, batch, muscle_images[i])
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

    torch.cuda.empty_cache()  # Clear GPU memory
    logger.warning(f"fitting done")
    return True
