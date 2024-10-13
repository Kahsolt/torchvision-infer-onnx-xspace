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
  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',
]
DEFAULT_MODEL = 'mobilenet_v3_small'


def softmax(x:ndarray) -> ndarray:
  x -= np.max(x, axis=-1, keepdims=True)
  expx = np.exp(x)
  return expx / np.sum(expx, axis=-1, keepdims=True)

def run_model(name:str, img:PILImage) -> Tuple[Pred, str]:
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


# TODO: add MAX_MODEL_CACHE
model_cache: Dict[str, Model] = {}

def get_model(name:str):
  if name in model_cache:
    model = model_cache[name]
    return model

  fp = MODEL_PATH / f'{name}.onnx'
  if fp.exists():
    model = ort.InferenceSession(fp)
    model_cache[name] = model
    return model

  fp = convert_model(name)
  model = ort.InferenceSession(fp)
  model_cache[name] = model
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
