import argparse
import PIL
import requests
import os
import torch
import numpy
import cv2 #opencv itself
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

def change_prompt(prompt, token, man):
  if man:
    return prompt.replace(token, "man with blue eyes")
  else:
    return prompt.replace(token, "woman with blue eyes")

def find_face(start_img):
  test_image = cv2.cvtColor(numpy.array(start_img), cv2.COLOR_RGB2BGR)
  grey = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(grey, 1.3, 5) # тут находим лицо
  return faces

def crop_face(image, faces, sec_num):
  faces_imgs = []
  image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
  for (x, y, w, h) in faces:
    cropped = image[y-sec_num:y + h+sec_num, x-sec_num:x + w+sec_num]  # тут обрезаем лицо + делаем края по 20 с каждой стороны
    faces_imgs.append(cropped)
  return faces_imgs

def to_pil(i):
  img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
  im_pil = Image.fromarray(img)
  return im_pil

def make_mask(c, sec_num):
  x = c.shape[0]
  y = c.shape[1]
  mask = cv2.rectangle(c, (0, 0), (x, y), (0, 0, 0), -1)  # так и должно быть!!!
  mask = cv2.rectangle(c, (sec_num, sec_num), (x-sec_num, y - sec_num), (255, 255, 255), -1)
  return mask

def generate_face(img, mask, prompt, seed, guidance_scale = 7.5, num_samples = 4):
  img = to_pil(img)
  mask = to_pil(mask)
  mask_image = mask.resize((512, 512))
  image = img.resize((512, 512))
  generator = torch.Generator(device="cuda").manual_seed(seed)  # change the seed to get different results
  images = pipe_inpaint(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
  ).images
  return images

def merge(original, face, faces_info, size_of_cropped, sec_num):
  face = resize_convert(face, size_of_cropped)  # только для первого лица
  x = faces_info[0, 0]
  y = faces_info[0, 1]
  original[y-sec_num:y-sec_num+face.shape[0], x-sec_num:x-sec_num+face.shape[1]] = face
  res = to_pil(original)
  return res

def resize_convert(pil_image, size_of_cropped):
  pil_image = pil_image.resize((size_of_cropped, size_of_cropped))  # без двойных скобок оно не работает!!!!
  face = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
  return face

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")

pipe_base = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")