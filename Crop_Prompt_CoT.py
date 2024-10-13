from transformers import AutoTokenizer
from PIL import Image, ImageFile
import math
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, device_map="cuda:2", cache_dir="/raid_sdd/zzy/19/.cache/huggingface/hub")

def crop_prompting(image_path, question_id, tmp_dir):
  crop_prompt = ''
  image = Image.open(image_path)
  #if image.mode == "P":
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

  query_list = [
    {'text': '<|im_start|>user\n'},
    {'image': image_path},
    {'text': 'Look carefully at this image and extract as much information as possible from the picture. There will be many pictures after this, but those pictures are all parts of the given wholistic picture. You are going to use more detailed information in those parts to assist your understanding of the wholistic picture so that you can perform better on visual tasks related to the wholistic picture.<|im_end|>\n'},
    {'text': '<|im_start|>assistant\nI have carefully analyzed the image you provided. As you mentioned, future images will provide more details, which can help refine the understanding of the holistic picture. I will keep this in mind and analyze each subsequent image with respect to the whole, aiming to improve performance in related visual tasks.<|im_end|>\n'},
    {'text': f'<|im_start|>user\nBelow are several detailed parts of this image. Their relative positions in the original image will be denoted as [`left`,`top`,`right`,`down`], in which the four variables are normalized to [0,1000). There are {cols}x{rows} parts in total.<|im_end|>\n'},
    {'text': f'<|im_start|>assistant\nThank you for providing the detailed parts of the image with positional information. I understand that the image is divided into a grid of {cols} columns and {rows} rows, with each part\'s position normalized in the range [0,1000) for the axes left, top, right, and down.\nHere\'s how I will approach the task:\n1. **Relative Position Analysis**: I will map each detailed part to its corresponding position within the overall image. This will help in maintaining the spatial context while analyzing the individual sections.\n2. **Feature Extraction**: For each part, I will extract key visual features, such as:\n- Objects or entities present in that section\n- Colors and texture patterns\n- Shapes and contours\n3. **Holistic Understanding**: As I analyze each part, I will integrate the findings to form a more complete understanding of the whole image. By correlating the visual details with their positions, I will ensure that spatial relationships are maintained to improve my overall understanding and performance in visual tasks.<|im_end|>\n'}
  ]

  try:
    os.makedirs(os.path.join(tmp_dir, question_id))
  except FileExistsError:
    pass

  for j in range(rows):
    for i in range(cols):
      crop_path = os.path.join(tmp_dir, question_id, f'crop_{j * cols + i + 1}.jpg')
      crops[j * cols + i].save(crop_path)
      query_list.extend([
        {'text': '<|im_start|>user\n'},
        {'image': crop_path},
        {'text': f'({i*1000/cols},{j*1000/rows}),({(i+1)*1000/cols},{(j+1)*1000/rows})<|im_end|>\n'},
        {'text': f'<|im_start|>assistant\nI have received the cropped image and its relative position in the larger image. The coordinates of this part are from ({i*1000/cols},{j*1000/rows}) to ({(i+1)*1000/cols},{(j+1)*1000/rows}). I will continue analyzing each crop and updating my understanding of the whole picture as new details emerge.<|im_end|>\n'}
      ])

  query_list.extend([
    {'text': '<|im_start|>user\nNow that all parts have been analyzed, review your previous observations and refine your understanding of the entire image to better handle visual-language tasks.\n- Use details from each part to support your description, and avoid repeating observations unless they provide new context.\n- Only describe elements that are directly visible in the image. Do not infer information that cannot be confirmed by visual evidence.\n- Verify that the details from each section are consistent and help refine your overall understanding of the image. Make sure all visual elements are consistent and aligned with the parts you\'ve seen.<|im_end|>\n'},
    {'text': '<|im_start|>assistant\nOkay, I will review and refine my understanding according to your instructions to improve my understanding of the image and better handle image-related tasks.<|im_end|>\n'}
  ])

  query = tokenizer.from_list_format(query_list)

  return query