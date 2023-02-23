import numpy as np
import torch
from wrn1d import WideResNet1d
from wrn2d import WideResNet2d
from wrn3d import WideResNet3d

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)


def get_wrn(
    input_shape,
    output_dim,
    output_shape,
    in_channels,
    device=None,
    depth=16,
    widen_factor=4,
    dropRate=0.0,
):
    """Init correct wrn
    """
    kwargs = {
        'depth': depth,
        'num_classes': output_dim,
        'input_shape': input_shape,
        'widen_factor': widen_factor,
        'dropRate': dropRate,
        'in_channels': in_channels,
        'output_shape': output_shape,
    }
    spacetime_dims = np.count_nonzero(np.array(input_shape)[[0, 2, 3]] != 1)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using WRN of dimension {spacetime_dims}")

    if spacetime_dims == 1:
        model_to_use = WideResNet1d
        # kwargs['dropRate'] = 0.4
    elif spacetime_dims == 2:
        model_to_use = WideResNet2d
    elif spacetime_dims == 3:
        model_to_use = WideResNet3d
    elif spacetime_dims == 0:  # Special case where we have channels only
        model_to_use = WideResNet1d
        kwargs['in_channels'] = 1
    else:
        raise NotImplementedError

    model = model_to_use(**kwargs).to(device)
    print(model)

    return model
