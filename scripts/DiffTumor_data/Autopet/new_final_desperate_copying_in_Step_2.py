import os
import shutil

# Define the directories
src_dir = 'imagesTs_Step_2_pet_ct_processed'
dst_dir = 'imagesTr_Step_2_pet_ct_processed'

# List directories in both source and destination
src_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
dst_dirs = sorted([d for d in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, d))])

# Determine the highest index in the destination directory
highest_index = 0
for dir_name in dst_dirs:
    index = int(dir_name.split('_')[-1])
    if index > highest_index:
        highest_index = index

# Copy and rename directories from source to destination
for dir_name in src_dirs:
    highest_index += 1
    new_name = '_'.join(dir_name.split('_')[:-1]) + f'_{highest_index}'
    shutil.copytree(os.path.join(src_dir, dir_name), os.path.join(dst_dir, new_name))

print("Directories copied and renamed successfully.")