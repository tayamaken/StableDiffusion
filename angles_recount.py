import argparse
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import requests
from transformers import AutoProcessor, CLIPModel
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import pathlib

urls_dict = {
    "lego": [
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/7/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/3/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/4/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/5/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/6/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/8/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/9/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/11/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/14/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/15/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/16/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/17/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/18/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/LegoStuff/--/tayamaken--LegoStuff/train/0/image/image.jpg'],
    "dasha": [
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/2/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/1/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/0/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/5/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/4/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/3/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/9/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/8/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/7/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/6/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/13/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/12/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/11/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/DashaStyleFinal/--/tayamaken--DashaStyleFinal/train/10/image/image.jpg'],
    "wimperg": [
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/2/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/3/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/14/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/13/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/12/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/11/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/10/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/9/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/8/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/7/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/6/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/5/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/1/image/image.jpg',
        'https://datasets-server.huggingface.co/assets/tayamaken/Vimperg/--/tayamaken--Vimperg/train/0/image/image.jpg']
}
def get_embeds(image: Image):
  image = pil_to_tensor(image)
  inputs = processor(images=image, return_tensors="pt")
  inputs['pixel_values'] = inputs['pixel_values']
  image_features = model.get_image_features(**inputs)
  return image_features
def unit_vectors(vectors):
  return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
def angles(a, b):
  a1 = unit_vectors(a)
  b1 = unit_vectors(b)
  return np.arccos(np.clip(np.dot(a1, b1.T), -1.0, 1.0))

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, choices=['lego', 'dasha', 'wimperg'], help='the name of the dataset')
parser.add_argument('--learning_rate', type=float, help="learning rate")
parser.add_argument('--max_train_steps', type=int, help="the amount of train steps")

args = parser.parse_args()

urls = urls_dict[args.dataset]
lr = args.learning_rate
ts = args.max_train_steps
dataset = args.dataset

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

target = []
for url1 in urls:
    image = Image.open(requests.get(url1, stream=True).raw)
    emb1 = get_embeds(image)
    target.append(emb1.detach().numpy()[0])

experiment_path = pathlib.Path(f'outputs/{dataset}/lr{lr}_ts{ts}')
experiment_path.mkdir(parents=True, exist_ok=True)
file_for_angles = experiment_path / "anglesNEW.txt"

for ns in [10, 30, 50, 75, 100]:
    for gs in [2.5, 7.5, 11, 15]:
      generated = []
      for num in range(16):
        url2 = f'https://storage.yandexcloud.net/dkozl-object-storage/{dataset}/lr{lr}_ts{ts}/ns{ns}_gs{gs}/num{num}.png'
        resp = requests.get(url2, stream=True)
        if not resp.ok:
          print('Failed to open', url2)
          continue
        image = Image.open(resp.raw)
        extrema = image.convert("L").getextrema()
        if extrema != (0,0):#это проверка на черное изображение
          emb2 = get_embeds(image)
          generated.append(emb2.detach().numpy()[0])
      t = np.array(target)
      g = np.array(generated)
      avgAngle = angles(t,g).mean()
      with open(file_for_angles, "a") as file:
          file.write(f'lr:{lr}\tts:{ts}\tns:{ns}\tgs:{gs}\timages:{len(g)}\tangle:{avgAngle}\n')
