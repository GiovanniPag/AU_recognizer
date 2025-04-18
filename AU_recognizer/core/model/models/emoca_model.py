import copy
import re
from pathlib import Path
from tkinter import StringVar, NSEW, EW

import numpy as np
import torch
from matplotlib import cm
from tqdm import auto

from AU_recognizer.ext_3dmm.emoca.datasets.ImageTestDataset import TestData
from AU_recognizer.ext_3dmm.emoca.models.DecaFLAME import FLAME_mediapipe
from AU_recognizer.ext_3dmm.emoca.utils.model import test, save_obj, save_images, save_codes
from AU_recognizer.ext_3dmm.emoca.utils.utility import load_model
from ..base_model_interface import BaseModelInterface
from ...user_interface import CustomFrame, ScrollableFrame, CustomCheckBox
from ...user_interface.widgets.complex_widget import RadioList
from ...util import nect_config, CONFIG, MODEL_FOLDER, MF_MODEL, MF_SAVE_IMAGES, MF_SAVE_CODES, MF_SAVE_MESH, \
    MF_FIT_MODE, i18n, R_DETAIL, R_COARSE, logger, P_PATH, F_OUTPUT, F_COMPARE, OBJ


class EmocaModel(BaseModelInterface):

    @staticmethod
    def get_models_list():
        models = [x for x in (Path(nect_config[CONFIG][MODEL_FOLDER]) / "EMOCA" / "models").iterdir() if x.is_dir()]
        return models

    @staticmethod
    def fit(fit_data, images_to_fit, project_data, progress_callback=None):
        # Implementazione fitting con Emoca
        try:
            logger.info(f"emoca fit of {images_to_fit}, with {fit_data}, and {project_data}")
            if isinstance(images_to_fit, list):
                images_to_fit = [str(image.resolve()) for image in images_to_fit]
            elif isinstance(images_to_fit, Path):
                images_to_fit = str(images_to_fit.resolve())
            else:
                logger.error(f"{images_to_fit} are not a list or a path")
                return False
            _project_name = str(project_data.sections()[0])
            project_path = Path(project_data[_project_name][P_PATH])
            path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER]) / "EMOCA" / "models"
            model_name = fit_data[MF_MODEL]
            output_folder = project_path / F_OUTPUT / model_name
            mode = fit_data[MF_FIT_MODE]

            torch.cuda.empty_cache()
            emoca, conf = (load_model
                           (path_to_models, model_name, mode))
            emoca.cuda()
            emoca.eval()

            dataset = TestData(images_to_fit, max_detection=20)

            for i in auto.tqdm(range(len(dataset))):
                try:
                    batch = dataset[i]
                except Exception as e:
                    logger.warning(f"error during fitting {dataset.imagepath_list[i]}: {e}")
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
                if progress_callback:
                    progress_callback((i + 1) / len(dataset))
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Fitting failed: {e}")
            return False  # Failure
        return True

    def get_mesh_from_params(self, params):
        # Ricostruzione mesh da parametri
        pass

    @staticmethod
    def au_difference(mesh_neutral, mesh_list, normalization_params, project):
        # Calcolo differenze
        _project_name = str(project.sections()[0])
        output_path = Path(project[_project_name][P_PATH]) / F_OUTPUT / F_COMPARE
        output_path.mkdir(parents=True, exist_ok=True)
        neutral_face_path = Path(mesh_neutral)
        neutral_obj = OBJ(filepath=neutral_face_path / "mesh_coarse.obj", generate_normals=False)
        n_verts = neutral_obj.get_vertices()
        neutral_pose = np.load(neutral_face_path / 'pose.npy')
        neutral_shape = np.load(neutral_face_path / 'shape.npy')
        # load emoca model
        path_to_models = Path(nect_config[CONFIG][MODEL_FOLDER]) / "EMOCA" / "models"
        model_name = "EMOCA_v2_lr_mse_20"
        mode = "detail"
        # Build a suffix string based on normalization options
        suffix = ""
        if normalization_params["pose"]:
            suffix += "_normPose"
        if normalization_params["identity"]:
            suffix += "_normIdentity"
        # 0) clear memory
        torch.cuda.empty_cache()  # Clear any cached GPU memory
        # 1) Load the model
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()
        for face_path in mesh_list:
            face_path = Path(face_path)
            face_obj = OBJ(filepath=face_path / "mesh_coarse.obj", generate_normals=False)
            f_exp = np.load(face_path / 'exp.npy')
            f_pose = np.load(face_path / 'pose.npy')
            f_shape = np.load(face_path / 'shape.npy')
            shape = torch.from_numpy(neutral_shape if normalization_params["pose"] else f_shape)
            exp = torch.from_numpy(f_exp)
            pose = torch.from_numpy(neutral_pose if normalization_params["identity"] else f_pose)
            shape = shape.unsqueeze(0)  # Shape: [1, 100]
            exp = exp.unsqueeze(0)  # Shape: [1, 50]
            pose = pose.unsqueeze(0)  # Shape: [1, 6]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            shape = shape.to(device)
            exp = exp.to(device)
            pose = pose.to(device)
            if not isinstance(emoca.deca.flame, FLAME_mediapipe):
                verts, landmarks2d, landmarks3d = emoca.deca.flame(shape_params=shape, expression_params=exp,
                                                                   pose_params=pose)
            else:
                verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = emoca.deca.flame(shape_params=shape,
                                                                                          expression_params=exp,
                                                                                          pose_params=pose)
            face_verts = verts.squeeze(0)  # Remove the batch dimension
            face_verts_np = face_verts.cpu().numpy()  # Convert PyTorch tensor to NumPy array
            diffs = np.linalg.norm(face_verts_np - n_verts, axis=1)  # Shape: [5023]
            normalized_diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))
            colormap = cm.get_cmap("plasma")  # Choose a colormap
            vertex_colors = colormap(normalized_diffs)[:, :3]  # RGB values for each vertex
            compare_obj = copy.deepcopy(face_obj)
            compare_obj.set_vertex_colors(vertex_colors)
            compare_obj.save(output_path / f"{face_path.name[:-2]}_with_heatmap{suffix}.obj")
            # save diff
            # Usa una regex per inserire un "_" tra lettere e numeri
            modified_name = re.sub(r'([A-Za-z]+)(\d)', r'\1_\2', face_path.name[:-2])
            np.save(output_path / f"{modified_name}_diffs{suffix}.npy", diffs)  # Save as binary

    def emoca_tag(self, diff_files, threshold, au_names):
        # Tagging topologia
        pass

    @staticmethod
    def get_ui_for_fit_data(master_widget) -> CustomFrame:
        frame = CustomFrame(master_widget)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        scrollFrame = ScrollableFrame(frame, orientation="horizontal")  # add a new scrollable frame
        scrollFrame.columnconfigure(0, weight=1)
        save_images_text = StringVar(value=i18n.entry_buttons["c_simages"])
        save_images = CustomCheckBox(master=scrollFrame, text="c_simages", textvariable=save_images_text,
                                     check_state=False)
        save_codes_text = StringVar(value=i18n.entry_buttons["c_scodes"])
        save_codes = CustomCheckBox(master=scrollFrame, text="c_scodes", textvariable=save_codes_text,
                                    check_state=True)
        save_mesh_text = StringVar(value=i18n.entry_buttons["c_smesh"])
        save_mesh = CustomCheckBox(master=scrollFrame, text="c_smesh", textvariable=save_mesh_text,
                                   check_state=True)
        fit_mode = RadioList(master=scrollFrame, list_title="mode_radio", default=R_DETAIL,
                             data=[R_DETAIL, R_COARSE])

        # Return frame and data extraction logic
        def get_data():
            return {
                MF_SAVE_IMAGES: save_images.get(),
                MF_SAVE_CODES: save_codes.get(),
                MF_SAVE_MESH: save_mesh.get(),
                MF_FIT_MODE: fit_mode.get_value()
            }

        def create_view():
            scrollFrame.grid(row=0, column=0, sticky=NSEW, padx=10, pady=5)
            # Checkboxes for options
            fit_mode.create_view()
            scrollFrame._check_scroll_necessity()

        def update_view():
            save_images.grid(row=0, column=0, sticky=EW, pady=5)
            save_codes.grid(row=1, column=0, sticky=EW, pady=5)
            save_mesh.grid(row=2, column=0, sticky=EW, pady=5)
            fit_mode.grid(row=3, column=0, sticky=EW, pady=5)
            scrollFrame._check_scroll_necessity()

        def update_language():
            save_images.update_language()
            save_codes.update_language()
            save_mesh.update_language()
            fit_mode.update_language()

        frame.get_data = get_data
        frame.update_language = update_language
        frame.create_view = create_view
        frame.update_view = update_view
        return frame
