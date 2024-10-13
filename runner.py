#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/13 

# 处理 onnx 模型的转换/运行

from time import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torchvision.models as M
import onnxruntime as ort
from onnxruntime import InferenceSession as Model
import numpy as np
from datetime import datetime
from numpy import ndarray
from PIL import Image
from PIL.Image import Image as PILImage

BASE_PATH = Path(__file__).parent
ASSET_PATH = BASE_PATH / 'assets'
MODEL_PATH = BASE_PATH / 'models'
DEMO_PATH  = BASE_PATH / 'demo'

Pred = Dict[str, float]
Stats = Dict[str, float]
npimg = ndarray

IMG_SIZE = 224
IMAGENET_MEAN = np.asarray([[[0.485, 0.456, 0.406]]], dtype=np.float32)
IMAGENET_STD  = np.asarray([[[0.229, 0.224, 0.225]]], dtype=np.float32)

SUPPORT_MODELS = [
  # .alexnet
  'alexnet',
  # .convnext
  'convnext_tiny',
  'convnext_small',
  'convnext_base',
  'convnext_large',
  # .densenet
  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',
  # .efficientnet
  "efficientnet_b0",
  'efficientnet_b1',
  'efficientnet_b2',
  'efficientnet_b3',
  'efficientnet_b4',
  'efficientnet_b5',
  'efficientnet_b6',
  'efficientnet_b7',
  'efficientnet_v2_s',
  'efficientnet_v2_m',
  'efficientnet_v2_l',
  # .googlenet
  'googlenet',
  # .inception
  'inception_v3',
  # .mnasnet
  'mnasnet0_5',
  'mnasnet0_75',
  'mnasnet1_0',
  'mnasnet1_3',
  # .mobilenet
  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',
  # .regnet
  'regnet_y_400mf',
  'regnet_y_800mf',
  'regnet_y_1_6gf',
  'regnet_y_3_2gf',
  'regnet_y_8gf',
  'regnet_y_16gf',
  'regnet_y_32gf',
  'regnet_y_128gf',
  'regnet_x_400mf',
  'regnet_x_800mf',
  'regnet_x_1_6gf',
  'regnet_x_3_2gf',
  'regnet_x_8gf',
  'regnet_x_16gf',
  'regnet_x_32gf',
  # .resnet
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',
  'wide_resnet50_2',
  'wide_resnet101_2',
  # .shufflenetv2
  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',
  # .squeezenet
  'squeezenet1_0',
  'squeezenet1_1',
  # .vgg
  'vgg11',
  'vgg11_bn',
  'vgg13',
  'vgg13_bn',
  'vgg16',
  'vgg16_bn',
  'vgg19',
  'vgg19_bn',
  # .vision_transformer
  'vit_b_16',
  'vit_b_32',
  'vit_l_16',
  'vit_l_32',
  'vit_h_14',
  # .swin_transformer
  'swin_t',
  'swin_s',
  'swin_b',
  'swin_v2_t',
  'swin_v2_s',
  'swin_v2_b',
  # .maxvit
  'maxvit_t',
]
DEFAULT_MODEL = 'mobilenet_v3_small'

MAX_MODEL_CACHE = 5

def now_ts() -> int:
  return int(datetime.now().timestamp())

def softmax(x:ndarray) -> ndarray:
  x -= np.max(x, axis=-1, keepdims=True)
  expx = np.exp(x)
  return expx / np.sum(expx, axis=-1, keepdims=True)

def run_model(name:str, img:PILImage) -> Tuple[Pred, str]:
  if not name:
    name = DEFAULT_MODEL
  if not img:
    return {}, 'Error: image is empty'
  
  ts_start = time()
  img = img.resize((IMG_SIZE, IMG_SIZE))
  im = np.asarray(img, dtype=np.float32) / 255.0
  im = (im - IMAGENET_MEAN) / IMAGENET_STD
  x = np.expand_dims(im.transpose(2, 0, 1), 0)
  ts_end = time()
  ts_preprocess = (ts_end - ts_start) * 1000

  model = get_model(name)
  input_name = model.get_inputs()[0].name
  output_name = model.get_outputs()[0].name
  ts_start = time()
  logits = model.run([output_name], {input_name: x})[0][0]
  ts_end = time()
  ts_infer = (ts_end - ts_start) * 1000

  ts_start = time()
  preds = softmax(logits)
  top5 = sorted([(p, idx) for idx, p in enumerate(preds)], reverse=True)[:5]
  results = {get_label(idx): p for p, idx in top5}
  ts_end = time()
  ts_postprocess = (ts_end - ts_start) * 1000

  info = f'[Time cost] preprocess: {ts_preprocess:.3f}ms, infer: {ts_infer:.3f}ms, postprocess: {ts_postprocess:.3f}ms'
  return results, info


model_cache: Dict[str, Tuple[Model, int]] = {}

def update_cache():
  if len(model_cache) < MAX_MODEL_CACHE:
    return
  else:
    sorted_cache = sorted(model_cache.items(), key=lambda x: x[1][1])
    print(sorted_cache)
    name = sorted_cache[0][0]
    del model_cache[name]
    return
  
def get_model(name:str):
  if name in model_cache:
    model, _ = model_cache[name]
    model_cache[name] = model, now_ts()
    return model

  fp = MODEL_PATH / f'{name}.onnx'
  if fp.exists():
    model = ort.InferenceSession(fp)
    update_cache()
    model_cache[name] = model, now_ts()
    return model

  fp = convert_model(name)
  model = ort.InferenceSession(fp)
  update_cache()
  model_cache[name] = model, now_ts()
  return model

def convert_model(name:str) -> Path:
  model = getattr(M, name)(pretrained=True).eval()
  MODEL_PATH.mkdir(exist_ok=True)
  fp = MODEL_PATH / f'{name}.onnx'
  torch.onnx.export(model, torch.randn(1, 3, 224, 224), fp, opset_version=11)
  return fp


label_en: List[str] = {}
label_cn: List[str] = {}

def get_label(idx:int) -> Tuple[str, str]:
  load_label()
  return f'{label_en[idx]} / {label_cn[idx]}'

def load_label():
  global label_en, label_cn
  if label_en and label_cn: return

  with open(ASSET_PATH  / 'label.txt', 'r', encoding='utf-8') as fh:
    label_en = [ln.strip() for ln in fh.read().strip().split('\n')]
  with open(ASSET_PATH  / 'label_cn.txt', 'r', encoding='utf-8') as fh:
    label_cn = [ln.strip() for ln in fh.read().strip().split('\n')]
  assert len(label_en) == len(label_cn) == 1000


if __name__ == '__main__':
  img = Image.open(DEMO_PATH / 'ILSVRC2012_val_00000081.png')
  preds, info = run_model(DEFAULT_MODEL, img)
  print('preds:', preds)
  print('info:', info)
