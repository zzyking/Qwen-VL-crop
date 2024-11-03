from PIL import Image, ImageFile
import math
import os
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_prompting(image_path, question_id, tmp_dir):
  crop_prompt = ''
  image = Image.open(image_path)
  # if image.mode == "P":
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

  if crop_width == 0 or crop_height == 0:
    return ''

  for row in range(rows):
    for col in range(cols):
      left = col * crop_width
      upper = row * crop_height
      right = (col + 1) * crop_width
      lower = (row + 1) * crop_height
      crop = image.crop((left, upper, right, lower))
      crops.append(crop)
  
  crop_prompt += f'The original image has a resolution of {width}x{height}. The following are parts of the original image with a resolution of {crop_width}x{crop_height} and their corresponding indices. There are {cols*rows} parts in total.'

  try:
    os.makedirs(os.path.join(tmp_dir, question_id))
  except FileExistsError:
    pass

 
  for j in range(rows):
    for i in range(cols):
      crop_path = os.path.join(tmp_dir, question_id, f'crop_{j * cols + i + 1}.jpg')
      if not os.path.exists(crop_path): crops[j * cols + i].save(crop_path)
      crop_prompt += f'<img>{crop_path}</img>this is the {j * cols + i + 1}th crop, with the upperleft coordinate of ({i * crop_width}, {j * crop_height}).'
  
  return crop_prompt