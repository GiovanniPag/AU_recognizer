from pathlib import Path
from omegaconf import OmegaConf

from ..models.DECA import DecaModule
from ..models.IO import locate_checkpoint
from AU_recognizer.core.util import asset


def load_model(path_to_models,
               run_name,
               stage
               ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    conf = replace_asset_dirs(conf, Path(path_to_models) / run_name)
    conf.coarse.checkpoint_dir = str(Path(path_to_models) / run_name / "coarse" / "checkpoints")
    conf.coarse.full_run_dir = str(Path(path_to_models) / run_name / "coarse")
    conf.coarse.output_dir = str(Path(path_to_models))
    conf.detail.checkpoint_dir = str(Path(path_to_models) / run_name / "detail" / "checkpoints")
    conf.detail.full_run_dir = str(Path(path_to_models) / run_name / "detail")
    conf.detail.output_dir = str(Path(path_to_models))
    deca = load_deca(conf,
                     stage
                     )
    return deca, conf


def replace_asset_dirs(cfg, output_dir: Path, ):
    asset_dir = get_path_to_models()

    for mode in ["coarse", "detail"]:
        cfg[mode].inout.output_dir = str(output_dir.parent)
        cfg[mode].inout.full_run_dir = str(output_dir / mode)
        cfg[mode].inout.checkpoint_dir = str(output_dir / mode / "checkpoints")

        cfg[mode].model.tex_path = str(asset_dir / "FLAME/texture/FLAME_albedo_from_BFM.npz")
        cfg[mode].model.topology_path = str(asset_dir / "FLAME/geometry/head_template.obj")
        cfg[mode].model.fixed_displacement_path = (
            str(asset_dir / "FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy"))
        cfg[mode].model.flame_model_path = str(asset_dir / "FLAME/geometry/generic_model.pkl")
        cfg[mode].model.flame_lmk_embedding_path = str(asset_dir / "FLAME/geometry/landmark_embedding.npy")
        cfg[mode].model.flame_mediapipe_lmk_embedding_path = str(
            asset_dir / "FLAME/geometry/mediapipe_landmark_embedding.npz")
        cfg[mode].model.face_mask_path = str(asset_dir / "FLAME/mask/uv_face_mask.png")
        cfg[mode].model.face_eye_mask_path = str(asset_dir / "FLAME/mask/uv_face_eye_mask.png")
        cfg[mode].model.pretrained_modelpath = str(asset_dir / "DECA/data/deca_model.tar")
        cfg[mode].model.pretrained_vgg_face_path = str(asset_dir / "FaceRecognition/resnet50_ft_weight.pkl")
        cfg[mode].model.emonet_model_path = ""

    return cfg


def get_path_to_models() -> Path:
    return asset("models")


def load_deca(conf,
              stage
              ):
    print(f"Taking config of stage '{stage}'")
    print(conf.keys())
    if stage is not None:
        cfg = conf[stage]
    else:
        cfg = conf
    cfg.model.resume_training = False

    checkpoint = locate_checkpoint(cfg, mode="best")
    if checkpoint is None:
        return None
    print(f"Loading checkpoint '{checkpoint}'")
    checkpoint_kwargs = {
        "model_params": cfg.model,
        "learning_params": cfg.learning,
        "inout_params": cfg.inout,
        "stage_name": "testing",
    }
    deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca
