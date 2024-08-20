from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import math
import tempfile
import os

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()

image_path = '/root/autodl-tmp/Qwen-VL/examples/0.JPG'
image = Image.open(image_path)
crops = []
crop_size = 448
width, height = image.size
width_crop = math.ceil(width / crop_size)
height_crop = math.ceil(height / crop_size)
# add padding
padded_image = Image.new("RGB",(width_crop*crop_size, height_crop*crop_size))
padded_image.paste(image,(0,0))
# cropping
for i in range(height_crop):
  for j in range(width_crop):
    crop = padded_image.crop((j*crop_size, i*crop_size, (j+1)*crop_size, (i+1)*crop_size))
    crops.append(crop)

with tempfile.TemporaryDirectory() as tmp:
  query_list = [
      {'image': image_path},
      {'text': f'This is the original image with a resolution of {width}x{height}. The following are parts of the original image with a resolution of {crop_size}x{crop_size} and their corresponding indices. There are {width_crop*height_crop} parts in total.'},
      {'text': 'From the information on that advertising board, what is the type of this shop?'}
  ]
  query = tokenizer.from_list_format(query_list)
  response, history = model.chat(tokenizer, query=query, history=None)
  print(response)
  for i in range(height_crop):
    for j in range(width_crop):
      crop_path = os.path.join(tmp, f'crop_{i * width_crop + j}.jpg')
      crops[i * width_crop + j].save(crop_path)
      query_list = [
        {'image': crop_path},
        {'text': f'this is the {i * width_crop + j + 1}th crop, with the index of ({i}, {j}).'},
        {'text': 'From the information on that advertising board, what is the type of this shop?'}
      ]
      query = tokenizer.from_list_format(query_list)
      response, history = model.chat(tokenizer, query=query, history=None)
      print(response)

  query_list = [
    {'text': 'from all the text history, can you answer the question?'},
    {'text': 'From the information on that advertising board, what is the type of this shop?'}
  ]
  query = tokenizer.from_list_format(query_list)
  response, history = model.chat(tokenizer, query=query, history=None)
  print(response)



  
