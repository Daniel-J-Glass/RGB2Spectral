import os
import re
import shutil

def split_files(directory, regex_A, regex_B):
  # Create the output directories
  dir_A = os.path.join(directory, "A")
  dir_B = os.path.join(directory, "B")
  dir_extra = os.path.join(directory, "Extra")
  os.makedirs(dir_A, exist_ok=True)
  os.makedirs(dir_B, exist_ok=True)
  os.makedirs(dir_extra, exist_ok=True)

  # Compile the regular expressions
  regex_A = re.compile(regex_A)
  regex_B = re.compile(regex_B)

  # Walk through the directory and its subdirectories
  for root, dirs, files in os.walk(directory):
    for file in files:
      # Check if the file name matches either regex_A or regex_B
      if regex_A.search(file):
        # Copy the file to dir_A
        src = os.path.join(root, file)
        dst = os.path.join(dir_A, os.path.basename(root) + "_" + file)
        shutil.copy(src, dst)
      elif regex_B.search(file):
        # Copy the file to dir_B
        src = os.path.join(root, file)
        dst = os.path.join(dir_B, os.path.basename(root) + "_" + file)
        shutil.copy(src, dst)
      else:
        # Copy the file to dir_extra
        src = os.path.join(root, file)
        dst = os.path.join(dir_extra, os.path.basename(root) + "_" + file)
        shutil.copy(src, dst)

if __name__=="__main__":
    split_files("../dataset/nirscene1/","rgb","nir")