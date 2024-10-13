#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/08 

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import gradio as gr
from runner import *

app = gr.Interface(
  fn=run_model,
  inputs=[
    gr.Dropdown(label='Model', value=DEFAULT_MODEL, choices=SUPPORT_MODELS, allow_custom_value=False, multiselect=False),
    gr.Image(label='Image', type='pil', image_mode='RGB', show_download_button=False),
  ],
  outputs=[
    gr.Label(label='Predict'),
    gr.Textbox(label='Message', max_lines=1),
  ],
  allow_flagging='never',
  examples=[[DEFAULT_MODEL, str(fp.relative_to(BASE_PATH))] for fp in DEMO_PATH.iterdir()],
  cache_examples=False,
  examples_per_page=4,
)

app.launch()
