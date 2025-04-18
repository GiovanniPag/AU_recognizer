import sys
from pathlib import Path

from ...models.IO import get_checkpoint_with_kwargs
from ...utils.other import class_from_str

try:
    from ...models.EmoNetModule import EmoNetModule
except ImportError as e:
    raise ImportError(
        f"Could not import EmoNetModule. EmoNet models will not be available. Make sure you pull the repository "
        f"with submodules to enable EmoNet.")


def emo_network_from_path(path):
    print(f"Loading trained emotion network from: '{path}'")

    def load_configs(run_path):
        from omegaconf import OmegaConf
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        if run_path != conf.inout.full_run_dir:
            conf.inout.output_dir = str(Path(run_path).parent)
            conf.inout.full_run_dir = str(run_path)
            conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
        return conf

    cfg = load_configs(path)

    if not bool(cfg.inout.checkpoint_dir):
        cfg.inout.checkpoint_dir = str(Path(path) / "checkpoints")

    checkpoint_mode = 'best'

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg,
                                                               checkpoint_mode=checkpoint_mode,
                                                               )
    checkpoint_kwargs = checkpoint_kwargs or {}
    if 'emodeca_type' in cfg.model.keys():
        module_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        module_class = EmoNetModule

    emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,
                                                      **checkpoint_kwargs)
    return emonet_module
