from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import math
import tempfile
import os

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
image_path = '/root/autodl-tmp/Qwen-VL/examples/sa_17012.jpg'
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

query_list = [
    {'text': f'The original image has a resolution of {width}x{height}. The following are parts of the original image with a resolution of {crop_size}x{crop_size} and their corresponding indices. There are {width_crop*height_crop} parts in total.'},
]

with tempfile.TemporaryDirectory() as tmp:
  for i in range(height_crop):
    for j in range(width_crop):
      crop_path = os.path.join(tmp, f'crop_{i * width_crop + j}.jpg')
      crops[i * width_crop + j].save(crop_path)
      query_list.extend([
        {'image': crop_path},
        {'text': f'this is the {i * width_crop + j + 1}th crop, with the index of ({i}, {j}).'},
      ])

  query_list.extend([
      {'image': image_path}, # Either a local path or an url
      {'text': "question: 'What is the brand of the bag?'"},
      {'text': "answer:"}
  ])

  query = tokenizer.from_list_format(query_list)

  inputs = tokenizer(query, return_tensors='pt')
  inputs = inputs.to(model.device)
  pred = model.generate(**inputs)
  response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
  print(response)
  # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
  image_boxed = tokenizer.draw_bbox_on_latest_picture(response)
  if image_boxed:
    image_boxed.save('2.jpg')
  else:
    print("no box")