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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
image_path = '/root/autodl-tmp/Qwen-VL/examples/caption_hard_hires.jpg'

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

# Select the appropriate scheme based on the ratio
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

query_list = [{'text': f'<context>The original image has a resolution of {width}x{height}. The following are parts of the original image with a resolution of {crop_width}x{crop_height} and their corresponding indices. There are {cols*rows} parts in total.\n '},]

with tempfile.TemporaryDirectory() as tmp:
  for j in range(rows):
    for i in range(cols):
      crop_path = os.path.join(tmp, f'crop_{j * cols + i + 1}.jpg')
      crops[j * cols + i].save(crop_path)
      query_list.extend([
        {'image': crop_path},
        {'text': f'this is the {j * cols + i + 1}th crop, with the upperleft coordinate of ({i * crop_width}, {j * crop_height}).\n'},
      ])

  query_list.extend([
      {'text': '</context>Information of different parts of the original image are within the <context> label and can be used as reference when answering the questions.\n'},
      {'image': image_path}, 
      {'text': 'You are a powerful image captioner. Instead of describing the imaginary content, only describ ing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Minimize aesthetic descriptions as much as possible.Describe the image in detail:'}, # Caption
      #{'text': 'What is the plane number? Answer:'} # VQA

  ])

  query = tokenizer.from_list_format(query_list)

  inputs = tokenizer(query, return_tensors='pt')
  inputs = inputs.to(model.device)
  pred = model.generate(**inputs)
  response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
  print(response)