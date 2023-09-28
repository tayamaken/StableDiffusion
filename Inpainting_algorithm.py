#проблема 1: не совсем понимаю, как установить нужные мне библиотеки и где это делать: в коде или снаружи его. и почему что-то подчеркивается
#проблема 2: не уверена, что все существующие тут библиотеки жизненно необходимы. не совсем знаю, как это проверить

pip install opencv-python
import argparse
import PIL
import requests
import torch
import numpy
import cv2 #opencv itself
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

parser = argparse.ArgumentParser()

parser.add_argument('prompt', type=str, help="your text description of the desired result")
parser.add_argument('embedding_path', type=str, help='the path to your embedding')
parser.add_argument('token', type=str, help="your token of the embedding")
parser.add_argument('if_man', type=bool, help="write 'True' if you want to draw a man, else 'False'")
parser.add_argument('path_for_result', type=str, help="a path to the folder(!) where to save the result")
parser.add_argument('--height', type=int, default=512, choices=[512, 720, 1080], help="image height")
parser.add_argument('--width', type=int, default=512, choices=[512, 720, 1080], help="image width")
parser.add_argument('--face_seed', default=0, type=int, help="a seed for generating a face, change to get other faces")

args = parser.parse_args()

prompt = args.prompt
token = args.token
embed_path = args.embed_path
man = args.if_man
height = args.height
width = args.width
face_seed = args.face_seed
path_for_result = args.path_for_result

def change_prompt(prompt, token, man):
  i = prompt.find('<')
  j = prompt.find('>')
  if man : return prompt.replace(token, "man with blue eyes")
  else: return prompt.replace(token, "woman with blue eyes")

def find_face(start_img):
  info = [6,4]
  test_image = cv2.cvtColor(numpy.array(start_img), cv2.COLOR_RGB2BGR)
  grey = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(grey, 1.3, 5) # тут находим лицо
  return faces

def crop_face(image, faces):
  faces_imgs = []
  image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
  for (x,y,w,h) in faces:
    cropped = image[y-20:y + h+20, x-20:x + w+20] #тут обрезаем лицо + делаем края по 20 с каждой стороны
    faces_imgs.append(cropped)
  return faces_imgs

def to_pil(i):
  img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
  im_pil = Image.fromarray(img)
  return im_pil

def make_mask(c):
  x = c.shape[0]
  y = c.shape[1]
  mask = cv2.rectangle(c, (0, 0), (x, y), (0,0,0), -1)
  mask = cv2.rectangle(c, (20, 20), (x-20, y -20), (255,255,255), -1)
  return mask

def generate_face(img, mask, prompt, seed, guidance_scale = 7.5, num_samples = 4):
  img = to_pil(img)
  mask = to_pil(mask)
  mask_image = mask.resize((512, 512))
  image = img.resize((512, 512))
  generator = torch.Generator(device="cuda").manual_seed(seed) # change the seed to get different results
  images = pipe_inpaint(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
  ).images
  return images

def merge(original, face, faces_info, size_of_cropped):
  face = resize_convert(face, size_of_cropped) #только для первого лица
  x = faces_info[0,0]
  y = faces_info[0,1]
  original[y-20:y-20+face.shape[0], x-20:x-20+face.shape[1]]= face
  res = to_pil(original)
  return res

def resize_convert(pil_image, size_of_cropped):
  pil_image = pil_image.resize((size_of_cropped, size_of_cropped)) #без двойных скобок оно не работает!!!!
  face = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
  return face

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# акак это вообще писать не в колабе??
!wget --no-check-certificate \
    https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/haarcascade_frontalface_default.xml \
    -O haarcascade_frontalface_default.xml

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained( #пусть пока будет так
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

pipe_base = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

prompt = change_prompt(prompt, token, man)

images = pipe_base(prompt, height, width, num_images_per_prompt=4, num_inference_steps=30, guidance_scale=11).images

the_best = 1 #пусть пока будет так, но когда-то как-то(я хз как) надо дать пользователю возможность выбора
img = images[the_best]
faces_info = find_face(img)
faces_arr = crop_face(img, faces_info)

#далее код, признающий существование только 1го лица
first_face = faces_arr[0]
mask = make_mask(first_face.copy())
size_of_cropped = (mask.shape[0])#красивее и адекватнее решения я не придумала

pipe_inpaint.load_textual_inversion(embed_path)
generated_faces = generate_face(first_face, mask, token ,2)

the_best_face = 1 #пусть пока будет так
result = merge(cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR),generated_faces[the_best_face], faces_info, size_of_cropped)

name_s = path_for_result / 'result.png'
result.save(str(name_s))

result #в колабе некоторые заморочки, насколько я понимаю, с показом изображения