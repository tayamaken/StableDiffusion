import argparse
import PIL
import requests
import os
import torch
import numpy as np
import cv2 #opencv itself
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForText2Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

def change_prompt(prompt, token, man):
  if man:
    return prompt.replace(token, "man with blue eyes")
  else:
    return prompt.replace(token, "woman with blue eyes")

def find_face(start_img):
  test_image = cv2.cvtColor(np.array(start_img), cv2.COLOR_RGB2BGR)
  grey = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(grey, 1.3, 5) # тут находим лицо
  return faces

def crop_face(image, faces, sec_num):
  faces_imgs = []
  image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

def merge(original, face, faces_info, sec_num):
  x = faces_info[0, 0]
  y = faces_info[0, 1]
  original[y-sec_num:y-sec_num+face.shape[0], x-sec_num:x-sec_num+face.shape[1]] = face
  res = to_pil(original)
  return res

def resize_convert(pil_image, size_of_cropped):
  pil_image = pil_image.resize((size_of_cropped, size_of_cropped))  # без двойных скобок оно не работает!!!!
  face = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
  return face

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def make_pyramid(face):
  k = int(0.15*face.size[0])
  width = face.size[0]
  x = np.linspace(-1, 1, width)
  y = np.linspace(-1, 1, width)
  xv, yv = np.meshgrid(x,y)
  f = abs(1 - np.maximum(abs(xv), abs(yv)))
  сorner = f[k, k]
  loc_max = f[int(width / 2), k - 1]
  f = np.where(f >= сorner, 1, f/loc_max)
  return f

def make_gradient(img_orig, result, face, faces_info, secret_num):
  ld_1 = img_orig.load()
  ld_2 = result.load()
  f = make_pyramid(face)
  for x in range(face.size[0]):
    for y in range(face.size[1]):
      k = f[x, y]
      color = []
      im_x = faces_info[0,0]- secret_num + x
      im_y = faces_info[0,1]- secret_num + y
      for i in range(3):
        color.append(int((1-k)*ld_1[im_x,im_y][i] + k*ld_2[im_x, im_y][i]))
      ld_1[im_x,im_y] = tuple(color)
  return img_orig

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")

pipe_base = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
