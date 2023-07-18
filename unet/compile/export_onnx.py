import torch
import torch.onnx
import os
import onnx
from onnxsim import simplify

from unet_model import UNet

folder = "./tmp"

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    torch.onnx.export(model, input, onnx_path, verbose=False, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

def onnx_simplify(onnx_path):
    # load your predefined ONNX model
    model = onnx.load(onnx_path)
    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
    onnx_path = 'tmp/unet_scale0.5.onnx'
    input = torch.randn(1, 3, 572, 572)
    pth_to_onnx(input, checkpoint, onnx_path)
    onnx_simplify(onnx_path)

