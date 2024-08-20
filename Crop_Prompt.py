from PIL import Image, ImageFile
import math
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_prompting(image_path, question_id, tmp_dir, crop_size = 448):
  crop_prompt = ''
  image = Image.open(image_path)
  crops = []
  width, height = image.size
  width_crop = math.ceil(width / crop_size)
  height_crop = math.ceil(height / crop_size)
  # add padding
  padded_image = Image.new("RGB",(width_crop*crop_size, height_crop*crop_size))
  padded_image.paste(image,(0,0))
  # cropping
  if (height_crop > 1 or width_crop > 1):
    for i in range(height_crop):
      for j in range(width_crop):
        crop = padded_image.crop((j*crop_size, i*crop_size, (j+1)*crop_size, (i+1)*crop_size))
        crops.append(crop)

    crop_prompt += f'The original image has a resolution of {width}x{height}. The following are parts of the original image with a resolution of {crop_size}x{crop_size} and their corresponding indices. There are {width_crop*height_crop} parts in total.'
    
    if not os.path.exists(os.path.join(tmp_dir, question_id)):
        os.makedirs(os.path.join(tmp_dir, question_id))

    for i in range(height_crop):
      for j in range(width_crop):
        crop_path = os.path.join(tmp_dir, question_id, f'crop_{i * width_crop + j + 1}.jpg')
        crops[i * width_crop + j].save(crop_path)
        crop_prompt += f'<img>{crop_path}</img>this is the {i * width_crop + j + 1}th crop, with the upperleft coordinate of ({i * crop_size}, {j * crop_size}).'

  return crop_prompt


'''
def crop_prompting(image_path, question_id, tmp_dir):
  crop_prompt = ''
  image = Image.open(image_path)
  crops = []
  width, height = image.size
  new_width = width // 2
  new_height = height // 2
  for i in range(2):
    for j in range(2):
      left = i * new_width
      upper = j * new_height
      right = (i + 1) * new_width
      lower = (j + 1) * new_height
      crop = image.crop((left, upper, right, lower))
      crops.append(crop)

  crop_prompt += f'The original image has a resolution of {width}x{height}. There are four crops of this image.'

  for i in range(2):
    for j in range(2):
      crop_path = os.path.join(tmp_dir, f'crop_{question_id}_{i * 2 + j + 1}.jpg')
      crops[i * 2 + j].save(crop_path)
      crop_prompt += f'<img>{crop_path}</img>this is the {i * 2 + j + 1}th crop, with the upper_left index of ({i*new_width}, {j*new_height}).'

  return crop_prompt
'''