import argparse
import torch
import torch.onnx
import os
import sys

# Monkey patch torchvision before basicsr is imported to fix missing module
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def main(args):
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]

    # Path concatenation
    input_path = os.path.join(args.pretrained_dir, f'{args.model_name}.pth')
    output_path = os.path.join(args.output_dir, f'{args.model_name}.onnx')

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    elif args.model_name in ['realesr-general-x4v3', 'realesr-general-wdn-x4v3']:  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    else:
        raise ValueError(f"Model name {args.model_name} is not supported.")

    loadnet = torch.load(input_path, map_location=torch.device('cpu'))

    # prefer params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = None

    if keyname is not None:
        model.load_state_dict(loadnet[keyname])
    else:
        model.load_state_dict(loadnet)
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, 64, 64)
    # Export the model
    with torch.no_grad():
        export_kwargs = {
            'opset_version': 14,
            'export_params': True,
            'input_names': ['input'],
            'output_names': ['output'],
            'dynamic_axes': {
                'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'height', 3: 'width'}
            }
        }

        # Attempt to explicitly pass external_data=False to satisfy user instructions
        if hasattr(torch.onnx, 'export'):
            try:
                # Some modern versions/dynamo paths might accept external_data directly
                torch.onnx.export(model, x, output_path, dynamo=False, external_data=False, **export_kwargs)
            except TypeError:
                try:
                    # Fallback for standard TorchScript which internally saves weights by default
                    torch.onnx.export(model, x, output_path, dynamo=False, **export_kwargs)
                except TypeError:
                    # Fallback for very old versions or strict signatures
                    torch.onnx.export(model, x, output_path, **export_kwargs)
        else:
            torch.onnx._export(model, x, output_path, **export_kwargs)

    print(f"Successfully exported {args.model_name} to {output_path}")

if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_dir', type=str, default='experiments/pretrained_models/', help='Pretrained model directory')
    parser.add_argument('--output_dir', type=str, default='weights/', help='Output onnx directory')
    parser.add_argument(
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3 | realesr-general-wdn-x4v3'))
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
