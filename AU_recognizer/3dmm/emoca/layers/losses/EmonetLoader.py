import torch
from ...utils.other import get_path_to_externals


def get_emonet(device=None, load_pretrained=True):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from AU_recognizer.external.emonet.models import EmoNet
    n_expression = 8
    # Create the model
    net = EmoNet(n_expression=n_expression).to(device)
    path_to_emonet = get_path_to_externals() / "emonet"
    state_dict_path = path_to_emonet / 'pretrained' / f'emonet_{n_expression}.pth'
    print(f'Loading the EmoNet model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)
    if not load_pretrained:
        print("Created an untrained EmoNet instance")
        net.reset_emo_parameters()
    net.eval()
    return net
