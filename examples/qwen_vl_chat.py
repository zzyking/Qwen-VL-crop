from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


sizes = [(448,448),(896,896),(1792,1792),(3584,3584),(7168,7168)]
# sizes = [(256,256),(128,128),(64,64),(32,32),(16,16),(8,8)]

name = 'clock2'

for target_size in sizes:
  query = tokenizer.from_list_format([
      {'image': f'/home/zzy/Qwen-VL-crop/examples/pad/{name}_{target_size[0]}x{target_size[1]}.jpg'}, # Either a local path or an url
      {'text': 'What is this?'},
  ]) 
  response, history = model.chat(tokenizer, query=query, history=None)
  print(target_size)
  print(response)

"""
query = tokenizer.from_list_format([
    {'image': f'/home/zzy/Qwen-VL-crop/examples/rescale/{name}.jpg'}, # Either a local path or an url
    {'text': 'What is this?'},
]) 
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

"""

"""
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('1.jpg')
else:
  print("no box")
"""