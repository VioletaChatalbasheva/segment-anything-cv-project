import os
import shutil

# Define the source and destination directories
current_dir = os.path.join(os.getcwd(), 'segment_anything', 'lower_body')
source_dir = 'images'
source_path = os.path.join(current_dir, source_dir)
destination_dir = 'cloth'
destination_path = os.path.join(current_dir, destination_dir)

# Create the destination directory if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Get all files in the source directory
files = os.listdir(source_path)

# Filter files that end with '_1.jpg'
filtered_files = [file for file in files if file.endswith('_1.jpg')]

# Move each filtered file to the destination directory
for file in filtered_files:
    shutil.move(os.path.join(source_path, file), os.path.join(destination_path, file))
