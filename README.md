# crop-n-caption
Two scripts useful for dealing with large image datasets of people. The first detects people and faces and crops photos. The second captions photos using BLIP2 and CLIP.

# Install

Open a windows command prompt, navigate to the folder in which you want to install, and then run the following commands. This will download the scripts into a subfolder called crop-n-caption, switch to that subfolder, and then run a windows script to install the necessary dependencies which will take several minutes.

```
git clone https://github.com/scottshireman/crop-n-caption
cd crop-n-caption
windows_setup.cmd
```

The windows script above will activate venv so you can get started right away. If you come back later in a new command prompt, you will need to manually start venv the following command:

```
activate_venv.bat
```


# Cropping images to people and/or faces
Assuming venv is active, you can see the parameters needed by typing
```
python --help extract.py
```

Which as of the time of writing this will return the following:
```
usage: extract.py [-h] [--img_dir IMG_DIR] [--out_dir OUT_DIR] [--person_probability PERSON_PROBABILITY]
                  [--face_probability FACE_PROBABILITY] [--append_folder_name] [--extract_people] [--extract_faces]
                  [--skip_multiples] [--no_compress] [--max_mp MAX_MP] [--quality QUALITY] [--overwrite] [--noresize]
                  [--training_size TRAINING_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Path to images
  --out_dir OUT_DIR     Path to folder for extracted images
  --person_probability PERSON_PROBABILITY
                        Minimum probability
  --face_probability FACE_PROBABILITY
                        Minimum probability
  --append_folder_name  Appends the folder names to the file name
  --extract_people      Extract images of people
  --extract_faces       Extract closeup face images
  --skip_multiples      Don't extract if multiple people or faces exist
  --no_compress         don't shrink large images or convert to webp. saves in original format.
  --max_mp MAX_MP       maximum megapixels (default: 1.5)
  --quality QUALITY     save quality (default: 95, range: 0-100, suggested: 90+)
  --overwrite           overwrite files in output directory
  --noresize            do not resize, just fix orientation
  --training_size TRAINING_SIZE
                        Size at which you intend to train. Puts smaller files to 'small' subfolder. 0 ignores.
```

