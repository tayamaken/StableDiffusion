import argparse
import itertools
import math
import os
import pathlib
import random
from io import BytesIO
from subprocess import getoutput

import accelerate
import numpy as np
import PIL
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from numpy import arccos, clip, dot
from numpy.linalg import norm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

# вынести из файла!!!!!!!!!!
# save_smth_path = '/content/SaverOfSmth' #папка для сохранения всякой всячины. нам не особо интересна(а коду интересна). Создать вне этого файла, название передать. Создать раз и навсегда
# save_smth = pathlib.Path(save_smth_path)
# save_smth.mkdir(exist_ok=True)

# datasetNum = 1 # каждого датасета свой номер, чтоб сохранялись результаты в разные папки
# save_root_path = f'/content/Saver{datasetNum}' #папка для сохранения нужной нам инфы. Создается для каждого датасета новая. Название передать
# save_root = pathlib.Path(save_root_path)
# save_root.mkdir(exist_ok=True)


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
wtt_dict = {
    "lego": "object",
    "dasha": "style",
    "wimperg": "object"
}
pt_dict = {
    "lego": "[lego_girl]",
    "dasha": "[dasha_style]",
    "wimperg": "[wimperg]"
}
it_dict = {
    "lego": "toy",
    "dasha": "childish",
    "wimperg": "architecture"
}
promts_dict = {
    "lego": "A photo of [lego_girl]",
    "dasha": "A picture in the style of [dasha_style]",
    "wimperg": "A photo of [wimperg]"
}
# promts_dict = {
#     "lego": "A [lego_girl] sitting on the bench in an autumn forest, morning lighting",
#     "dasha": "A cute cat with a green bow sitting on the windowsill, sunset, [dasha_style]",
#     "wimperg": "A huge window with [wimperg] on a wall of an old church"
# }

parser = argparse.ArgumentParser()

parser.add_argument('--save_smth_path', type=str, help="a path to the folder for all we don't need")
parser.add_argument('--save_root_path', type=str, help="a path to the folder for all we need")
parser.add_argument('--dataset', type=str, choices=['lego', 'dasha', 'wimperg'], help='the name of the dataset')
parser.add_argument('--learning_rate', type=float, help="learning rate")
parser.add_argument('--max_train_steps', type=int, help="the amount of train steps")
parser.add_argument('--num_samples', type=int, help="how many images we generate")

args = parser.parse_args()

save_root_path = args.save_root_path
save_smth_path = args.save_smth_path
urls = urls_dict[args.dataset]
what_to_teach = wtt_dict[args.dataset]
placeholder_token = pt_dict[args.dataset]
initializer_token = it_dict[args.dataset]
prompt = promts_dict[args.dataset]
learning_rate = args.learning_rate
max_train_steps = args.max_train_steps
num_samples = args.num_samples

#creating folders and files
save_path = pathlib.Path(save_smth_path)/f"lr{learning_rate}_ts{max_train_steps}"

experiment_path = pathlib.Path(save_root_path)/f'lr{learning_rate}_ts{max_train_steps}'
experiment_path.mkdir()

file_for_angles = pathlib.Path(save_root_path) / f'lr{learning_rate}_ts{max_train_steps}' / "angles.txt"

# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"


#Downloading the dataset
def download_image(url):
  try:
    response = requests.get(url)
  except:
    return None
  return Image.open(BytesIO(response.content)).convert("RGB")

images = list(filter(None,[download_image(url) for url in urls]))

if not os.path.exists(save_path):
  os.mkdir(save_path)
[image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]

images = []
for file_path in os.listdir(save_path):
  try:
      image_path = os.path.join(save_path, file_path)
      images.append(Image.open(image_path).resize((512, 512)))
  except:
    print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")


#Setting up the prompt templates for training
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

#Setting up the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
# Loading the tokenizer and add the placeholder token as a additional special token.
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

# Adding the placeholder token in tokenizer
num_added_tokens = tokenizer.add_tokens(placeholder_token)
if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {placeholder_token}. Please pass a different "
                     f"`placeholder_token` that is not already in the tokenizer.")

token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
# Checking if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")
initializer_token_id = token_ids[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# Loading the SD model
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)

#A little magic with tokens' embeds
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

def freeze_params(params):
    for param in params:
        param.requires_grad = False

# Freeze vae and unet
freeze_params(vae.parameters())
freeze_params(unet.parameters())
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)

#Creating the Dataset and Dataloader
train_dataset = TextualInversionDataset(
      data_root=save_path,
      tokenizer=tokenizer,
      size=vae.sample_size,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach,
      center_crop=False,
      set="train",
)
def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

#Creating noise_scheduler for training
noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

#Setting up all training args
hyperparameters = {
    "learning_rate": learning_rate,
    "scale_lr": True,
    "save_steps": 2000,
    "max_train_steps":max_train_steps,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": save_path}

#Training function
logger = get_logger(__name__)
def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict,save_path)
def training_function(text_encoder, vae, unet):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hyperparameters["mixed_precision"]
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hyperparameters["save_steps"] == 0:
                    save_path = output_dir/f"learned_embeds-step-{global_step}.bin"
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        save_path = experiment_path/ f"learned_embeds.bin"
        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

#Training!!
accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet), num_processes=1)

for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
  if param.grad is not None:
    del param.grad  # free some memory
  torch.cuda.empty_cache()
del text_encoder
del vae
del unet
del tokenizer

GENERAION_DEVICE = 'cuda'

#Setting up the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],
    scheduler=DPMSolverMultistepScheduler.from_pretrained(hyperparameters["output_dir"], subfolder="scheduler"),
    torch_dtype=torch.float16,
).to(GENERAION_DEVICE)


#Getting the average angle
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(GENERAION_DEVICE)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
def get_embeds(image: Image):
  image = pil_to_tensor(image).to(GENERAION_DEVICE)
  inputs = processor(images=image, return_tensors="pt")
  inputs['pixel_values'] = inputs['pixel_values'].to(GENERAION_DEVICE)
  image_features = model.get_image_features(**inputs)
  return image_features
def angle(emb1, emb2):
  u = np.squeeze(np.asarray(emb1.detach().cpu()))
  v = np.squeeze(np.asarray(emb2.detach().cpu()))
  c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
  angle = arccos(clip(c, -1, 1)) # if you really want the angle
  return(angle)

#Saving all the received images
for ns in [10, 30, 50, 75, 100]:
    for gs in [2.5, 7.5, 11, 15]:
        paths = []
        all_images = []
        # SD works only for 2 images max. Generate 64 images:
        for _ in range(8):
            images = pipe([prompt] * num_samples, num_inference_steps=ns, guidance_scale=gs).images
            all_images.extend(images)
        folder_name = f'ns{ns}_gs{gs}'
        my_path = experiment_path / folder_name
        my_path.mkdir(exist_ok=True)

        for i in range(len(all_images)):
            name_s = my_path / f'num{i}.png'
            all_images[i].save(str(name_s))
            paths.append(name_s)

        anglesSum = 0
        url1 = urls[0]
        image = Image.open(requests.get(url1, stream=True).raw)
        emb1 = get_embeds(image)
        for i in range(len(all_images)):
            url2 = paths[i]
            image2 = Image.open(url2)
            emb2 = get_embeds(image2)
            anglesSum += angle(emb1, emb2)
        avgAngle = anglesSum/len(all_images)
        with open(file_for_angles, "a") as file:
            file.write(f'lr:{learning_rate}\tts:{max_train_steps}\tns:{ns}\tgs:{gs}\tangle:{avgAngle}\n')

