from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import math
import tempfile
import os

import sys
sys.path.append('/root/autodl-tmp/Qwen-VL')
from Crop_Prompt_No_Padding import crop_prompting

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
image_path = '/root/autodl-tmp/Qwen-VL/examples/caption_easy_lowres.jpg'

image = Image.open(image_path)
if image.mode == "P":
  image = image.convert('RGB')
crops = []
width, height = image.size

# width-to-height ratio
ratio = width / height
# cropping schemes
schemes = {
  (2, 2): (0.75, 1.33), # Between 3:4 and 4:3
  (1, 2): (0.5, 0.75), # Between 1:2 and 3:4
  (2, 1): (1.33, 2), # Between 4:3 and 2:1
  (1, 3): (0.33, 0.5), # Between 1:3 and 1:2
  (3, 1): (2, 3), # Between 2:1 and 3:1
  (1, 4): (0, 0.33), # Between 1:inf and 1:3
  (4, 1): (3, float('inf')) # 3:1 and beyond
}

# Select the appropri ate scheme based on the ratio
selected_scheme = None
for scheme, (min_ratio, max_ratio) in schemes.items():
  if min_ratio <= ratio < max_ratio:
    selected_scheme = scheme
    break

cols, rows = selected_scheme
crop_width = int(width // cols)
crop_height = int(height // rows)

for row in range(rows):
  for col in range(cols):
    left = col * crop_width
    upper = row * crop_height
    right = (col + 1) * crop_width
    lower = (row + 1) * crop_height
    crop = image.crop((left, upper, right, lower))
    crops.append(crop)

query_list = [{'text': f'<context>\nThe entire image has a resolution of {width}x{height}. The following are parts of the entire image with the same size. Their relative positions in the original image will be denoted as [`left`,`top`,`right`,`down`], in which the four variables are normalized to [0,1000). There are {cols}x{rows} parts in total.\n'}]

with tempfile.TemporaryDirectory() as tmp:
  for j in range(rows):
    for i in range(cols):
      crop_path = os.path.join(tmp, f'crop_{j * cols + i + 1}.jpg')
      crops[j * cols + i].save(crop_path)
      query_list.extend([
        {'image': crop_path},
        {'text': f'({i*1000/cols},{j*1000/rows}),{(i+1)*1000/cols},{(j+1)*1000/rows}]\n'}
      ])

  query_list.extend([
      {'text': f'</context>\n\nThe information above is provided as context to assist your understanding. Picture {cols*rows+1} is the entire image and your task is as follows:\n'},
      {'image': image_path},
      {'text': '\n<instructions>\n1. Examine the full image carefully.\n2. Use the crop information as background knowledge ONLY.\n3. Generate ONE comprehensive caption for the entire image shown.\n4. Focus on describing the overall scene and main elements visible.\n5. Do not caption individual crops or break your response into sections.\n</instructions>\n\nDescribe this image as detailed as possible:'}
       # Caption
      #{'text': 'What is the plane number? Answer:'} # VQA

  ])

  query = tokenizer.from_list_format(query_list)
  #print(query)
  #tokens = tokenizer.encode(query, add_special_tokens=True)
  #token_count = len(tokens)

  #print(token_count)
  response, history = model.chat(tokenizer, query=query, history=None)

  print(response)