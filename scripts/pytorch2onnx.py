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

    loadnet = torch.load(args.input, map_location=torch.device('cpu'))

    # prefer params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    model.load_state_dict(loadnet[keyname])
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

        # Check if we should pass external_data (available in torch 2.1+ via dynamo, but user asked specifically to set external_data to false, so we pass it if we use dynamo, or we stick to standard export where it's False by default. We can use dynamo=False to use TorchScript exporter which exports weights internally by default.)
        if hasattr(torch.onnx, 'export'):
            try:
                torch.onnx.export(model, x, args.output, dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(model, x, args.output, **export_kwargs)
        else:
            torch.onnx._export(model, x, args.output, **export_kwargs)

    print(f"Successfully exported {args.model_name} to {args.output}")

if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan-x4.onnx', help='Output onnx path')
    parser.add_argument(
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3 | realesr-general-wdn-x4v3'))
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
