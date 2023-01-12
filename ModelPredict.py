import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from PIL import Image
import imageio
from scipy import ndimage
from skimage import exposure

# Set random seed for reproducibility
manual_seed = 999
#manual_seed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

MODEL_INPUT_SIZE = (2048,1024)

def adjust_contrast(image, target_image):
    # Convert images to arrays
    image = np.array(image)
    target_image = np.array(target_image)

    # Calculate the mean values of the images
    image_mean = image.mean()
    target_mean = target_image.mean()

    # Calculate the standard deviations of the images
    image_std = image.std()
    target_std = target_image.std()

    # Calculate the scaling factor for the contrast adjustment
    scale_factor = target_std / image_std

    # Adjust the contrast of the image
    adjusted_image = (image - image_mean) * scale_factor + target_mean

    # Clip the image values to be within the range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # Convert the image back to an Image object
    adjusted_image = Image.fromarray(adjusted_image)

    return adjusted_image

def adjust_brightness(image, target_image):
    # Convert images to arrays
    image = np.array(image)
    target_image = np.array(target_image)

    # Calculate the mean values of the images
    image_mean = image.mean()
    target_mean = target_image.mean()

    # Calculate the scaling factor for the brightness adjustment
    scale_factor = target_mean / image_mean

    # Adjust the brightness of the image
    adjusted_image = image * scale_factor

    # Clip the image values to be within the range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # Convert the image back to an Image object
    adjusted_image = Image.fromarray(adjusted_image)

    return adjusted_image

def apply_model_to_image(model, input_path, output_path, tile_size=256, overlap = 32,tiling_opacity=70, blur_sigma = 64):
    # Load the image and get its size
    image = Image.open(input_path).convert("RGB")

    width, height = image.size
    if tile_size>width:
        tile_size=width
    if tile_size>height:
        tile_size=height

    # Initialize an output image with the same size as the input
    output_image = Image.new("RGB", (width, height))

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    image = transform(image).unsqueeze(0).to(device)

    #scale temp image to max model input size
    temp_image = image.clone()
    temp_image = F.interpolate(temp_image, size=MODEL_INPUT_SIZE, mode='bilinear', align_corners=False)

    #infer with model
    with torch.no_grad():
        comp_output = model((temp_image,torch.tensor([0])))

    #scale up model output and save to variable comp_image
    comp_output = comp_output.squeeze(0).cpu()
    comp_output = comp_output.permute(1, 2, 0).numpy()
    comp_output = (comp_output + 1) / 2
    comp_output = (comp_output * 255).astype(np.uint8)
    comp_image = Image.fromarray(comp_output)
    comp_image = comp_image.resize((width, height))
    comp_image.save("Comp.png")
    last_tile = None
    output_images = {}
    for i in range(0, width-tile_size+overlap, tile_size-overlap):
        for j in range(0, height-tile_size+overlap, tile_size-overlap):
            # Shift the tile position if it goes out of bounds
            if i+tile_size>width:
                i= width-tile_size
            if j+tile_size>height:
                j = height-tile_size

            x=i
            y=j
            
            if last_tile == (x,y):
                continue 

            # Extract the current tile
            tile_image = image[:,:,y:y+tile_size,x:x+tile_size]

            # Make prediction
            with torch.no_grad():
                output = model((tile_image,torch.tensor([0])))


            # Postprocess
            output = output.squeeze(0).cpu()
            output = output.permute(1, 2, 0).numpy()
            output = (output + 1) / 2
            output = (output * 255).astype(np.uint8)

            output = Image.fromarray(output)

            # use a corresponding tile of comp_image to adjust white balance of the model output tile
            # This is to use the correct white balance from the smaller comp_image to make the tile \
            #   white balances more consistent across the final image 
            comp_tile = comp_image.crop((x, y, x+tile_size, y+tile_size))
            output = adjust_contrast(output,comp_tile)
            output = adjust_brightness(output,comp_tile)

            # Tile masking
            edge = int(tile_size/4)
            mask = np.ones(output.size)*255

            # Creating masks for all tile positions
            # X
            print(f"{x},{y}")
            if not x==0:# not left
                print("Not left")
                mask[:,:edge]=0
            if not x>=width-tile_size:# not right
                print("Not right")
                mask[:,-edge:]=0
            # Y
            if not y==0:# not top
                print("Not top")
                mask[:edge,:]=0
            if not y>=height-tile_size:# not bottom
                print("Not bottom")
                mask[-edge:,:]=0
            
            mask = ndimage.gaussian_filter(mask, sigma=blur_sigma)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask, mode="L")

            # Paste the processed tile into the output image with the mask
            output_image.paste(output, (x,y),mask)

            # Save the output image and output image without mask in the dictionary (possible no longer needed)
            output_images[(x,y)] = (output, mask)

            #preventing accidental duplicates
            last_tile = (x,y)

            #Intermediate image save for progress tracking
            output_image.save("./Temp.jpg")
        if (x,y) == (width,height):
            break

    output_image.save(output_path)

def main():
    # Load model
    model = torch.load("model.pth", map_location=device)
    model.eval()

    tile_size = 1024
    image_path = "../RGB/DSC00563.png"
    output_path = f"./1-IR{tile_size}.png"
    apply_model_to_image(model,image_path,output_path,tile_size, overlap=768,blur_sigma=64)

    return

if __name__ =="__main__":
    main()