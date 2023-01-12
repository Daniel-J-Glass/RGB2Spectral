from glob import glob

import random
from PIL import Image

import re
import os
import shutil

def generate_crops(image, num_crops, crop_size):
    """
    Generates evenly spaced crops from an image.
    
    Parameters:
    - image (str): The file path of the image to be cropped.
    - num_crops (int): The number of crops to generate.
    - crop_size (tuple): The size of the crops to generate (x, y).
    
    Returns:
    - List[Image]: A list of PIL.Image objects representing the crops.
    """
    # Open the image
    im = Image.open(image)
    
    # Get the size of the image
    width, height = im.size
    
    # Calculate the horizontal and vertical spacing between crops
    h_spacing = (width - crop_size[0]) // (num_crops - 1)
    v_spacing = (height - crop_size[1]) // (num_crops - 1)
    
    # Initialize an empty list to store the crops
    crops = []
    
    # Generate the crops
    for i in range(num_crops):
        for j in range(num_crops):
            left = i * h_spacing
            top = j * v_spacing
            right = left + crop_size[0]
            bottom = top + crop_size[1]
            crops.append(im.crop((left, top, right, bottom)))
        
    return crops

def process_files(root_dir, func, num_crops, crop_size):
    """
    Processes all files in the lowest level folders of a directory and saves the results in a new directory.
    
    Parameters:
    - root_dir (str): The root directory to search.
    - func (callable): The function to apply to the files.
    - num_crops (int): The number of crops to generate for each file.
    - crop_size (tuple): The size of the crops to generate (x, y).
    """
    # Create the destination directory
    dest_dir = root_dir + '_processed'
    os.makedirs(dest_dir, exist_ok=True)
    
    # Initialize a list of directories to process
    dirs_to_process = [root_dir]
    
    # Iterate over the directories to process
    while dirs_to_process:
        # Get the next directory to process
        curr_dir = dirs_to_process.pop()
        
        file_num = 0
        # Iterate over the items in the directory
        for item in os.listdir(curr_dir):
            # Get the full path of the item
            item_path = os.path.join(curr_dir, item)
            
            # If the item is a directory, add it to the list of directories to process
            if os.path.isdir(item_path):
                dirs_to_process.append(item_path)
            
            # If the item is a file, process it
            elif os.path.isfile(item_path):
                # Apply the function to the file
                crops = func(item_path, num_crops, crop_size)
                
                # Save the crops in the destination directory
                for crop in crops:
                    # Construct the destination path for the crop
                    rel_path = os.path.relpath(item_path, root_dir)
                    file_path = os.path.basename(rel_path)
                    file = os.path.splitext(file_path)
                    file_name = file[0]
                    file_ext = file[1]
                    
                    rel_dir = os.path.dirname(rel_path)
                    dir_name = os.path.join(dest_dir,rel_dir)
                    os.makedirs(dir_name,exist_ok=True)

                    # Replace the invalid characters in the file name with underscores
                    file_name = os.path.join(dir_name,re.sub(r'[\\/:"*?<>|]', '_', file_name))
                    crop_path = os.path.join(file_name + f'_{file_num}{file_ext}')
                    crop.save(crop_path)
                    file_num+=1


if __name__=="__main__":
    dataset = r"..\dataset\nirscene_split"
    crop_size = (512,512)
    num_crops = 2 #numcrops output is squared
    process_files(dataset,generate_crops,num_crops,crop_size)
