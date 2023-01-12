import numpy as np
import cv2
import os
import colour
from typing import List,Tuple

def filter_wavelengths(input_path: str, output_path: str, wavelength_ranges: List[Tuple[int, int]]):
    """returns an image with colors filtered to be within the list of light wavelengths ranges\
        by converting the rgb value to a spectral representation and filtering based on the \
            wavelengths passed.

    Args:
        input_path (str): file path for image input
        output_path (str): file path for image output
        wavelength_ranges (list): list of accepted wavelength ranges for the image
    """
    # Load image
    image = cv2.imread(input_path)
    filtered_sd = colour.image.rgb_to_sd(image, wavelength_range=wavelength_ranges)
    
    # Convert the filtered spectral distribution data back to an RGB image
    filtered_image = colour.sd_to_rgb(filtered_sd)

    # Save image
    cv2.imwrite(output_path, filtered_image)
    
    return

if __name__=="__main__":
    input_path = "../RGB/2.jpg"
    output_path = "../filtered.jpg"
    image = cv2.imread(input_path)
    print(image.shape)
    # colour.convert()
    new_image = colour.recovery.RGB_to_spectral_Smits1999()
    print(new_image.shape)
    # print(rgb_to_wavelength((1,0,0)))
    # filter_wavelengths(input_path, output_path, [(200, 400)])
