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


# Crop images of people and/or faces from photos
crop.py uses the imageai library to detect people and the mediapipe library to detect faces in photos. It then crops people/faces and saves them as new images. It will fix orientation issues if exif data is available and will ensure output image aspect ratios are between 1:4 to 4:1. The resuling images work very well in Stable Diffusion trainers such as [EveryDream2](https://github.com/victorchall/EveryDream2trainer).

#  Usage
Assuming venv is active, you can see the parameters needed by typing
```
python crop.py --help
```

Which as of the time of writing this will return the following:
```
usage: crop.py [-h] [--img_dir IMG_DIR] [--out_dir OUT_DIR] [--person_probability PERSON_PROBABILITY]
               [--face_probability FACE_PROBABILITY] [--append_folder_name] [--crop_people] [--crop_faces]
               [--skip_multiples] [--no_compress] [--max_mp MAX_MP] [--quality QUALITY] [--overwrite] [--no_transpose]
               [--training_size TRAINING_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Path to input images. (default: 'input')
  --out_dir OUT_DIR     Path to output folder for cropped images. (default: 'output')
  --person_probability PERSON_PROBABILITY
                        Minimum probability threshold for detecting peoeple. Lower means more false positives, higher
                        means more false negatives. (default: 50).
  --face_probability FACE_PROBABILITY
                        Minimum probability threshold for detecting faces. Lower means more false positives, higher
                        means more false negatives. (default: 50).
  --append_folder_name  Prepends to output filenames the names of all subfolders the input file is nested in,
                        delimitted by '_' Helpful for some large datasets with lots of meaningful subfolders that can
                        be used later for caption tags.
  --crop_people         Crop images of people from img_dir file and save as new images in in out_dir
  --crop_faces          Crop images of faces from img_dir file and save as new images in out_dir
  --skip_multiples      Some images have lots of people/faces and by default all will be extracted. For very large
                        datasets that might mean a lot of false negatives. Set this option to ignore any input image
                        that has multiple people or faces (evaluated seperately).
  --no_compress         By default cropped output images will be written as compressed webp files. Use this flag to
                        instead save them in the same format as the original input image.
  --max_mp MAX_MP       Maximum megapixels to save cropped output images. Larger images will be shrunk to this value.
                        Images with not be resized at all if --no_compress flag is set. (default: 1.5)
  --quality QUALITY     Webp quality level to save cropped output images. Will not apply if --no_compress flag is set.
                        (default: 95)
  --overwrite           Overwrite files in output directory if they already exist
  --no_transpose        By default images will be transposed to proper orientation if exif data on the roper
                        orientation exists even if --no_compress is specified. Set this flag to disable.
  --training_size TRAINING_SIZE
                        The resolution at which you intend to train. Cropped images that are smaller than that will be
                        written to 'small' subfolder created in the output folder. Specify 0 to ignore. (default: 512)
```

The simplest usage would be to run the following command:
```
python crop.py --crop_people --crop_faces
```
This will scan all files in the ```input``` folder and all sufolders to find anything the models is at least 50% confident is a person or face, crop those people and faces, and save them as new images in the ```output``` folder as webp files. Any files smaller than 262,144 pixels needed to train at 512 resoltuon will instead be written to the ```small``` subfolder inside the ```output``` folder.


# Caption images of people using BLIP2 and CLIP
caption.py uses BLIP2 and CLIP to create captions of people. It is not intended as a general purpose captioner, but for the very specific purpose of captioning images of people. For a more general purpose trainer I recommend [captionr](https://github.com/theovercomer8/captionr) or the caption.py script that is included in [EveryDream2](https://github.com/victorchall/EveryDream2trainer).

caption.py will first use BLIP2 to generate a good base caption such as 'a woman in a coat and scarf posing in the park' and then it will use CLIP generate two tags, one descriptive of the style of the photo such as 'an outdoor photo' and the other descriptive of the person's emotion/facial expression such as 'happy'. It then writes a full caption to a txt or yaml file. The possible photo styles (mediums.txt) and emotion/expressions (emotions.txt) are fully customizable by editing the appriptiate txt file in the ```data``` folder.

The end result in this example could be a text file as follows (default):
```
a woman in a coat and scarf posing in the park, an outdoor photo, happy
```

or it could be a yaml file as follows (recomended if your trainer supports it and you use folders to organize images):
```
main prompt: a woman in a coat and scarf posing in the park
tags
  - tag: an outdoor photo
  - tag: happy
```

The advantage of a yaml file is that some trainers like [EveryDream2](https://github.com/victorchall/EveryDream2trainer) allow you to use global.yaml or local.yaml files at the subfolder to add further tags easily to groups of photos which gives a great deal of control to quickly add tags to refine your training.

The script can also replace generic terms like 'a woman' or 'a lovely woman' with a specific term like 'jane doe' for fine-tuning Stable Diffusion models.

I've found captions formatted this way work well in Stable Diffusion trainers such as EveryDream2. In my tests  improve slightly loss/val but also 

#  Usage
Assuming venv is active, you can see the parameters needed by typing
```
python caption.py --help
```
Which as of the time of writing this will return the following:
```
usage: caption.py [-h] [--img_dir IMG_DIR] [--blip_model BLIP_MODEL] [--clip_model CLIP_MODEL] [--force_cpu]
                  [--max_new_tokens MAX_NEW_TOKENS] [--find [FIND]] [--replace [REPLACE]] [--yaml]
                  [--mediums [MEDIUMS]] [--emotions [EMOTIONS]]

optional arguments:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Path to input images. (default: 'output')
  --blip_model BLIP_MODEL
                        BLIP2 moodel from huggingface. (default: salesforce/blip2-opt-6.7b)
  --clip_model CLIP_MODEL
                        CLIP model. (default: ViT-L-14/openai)
  --force_cpu           Force using CPU for BLIP even if GPU is available. You need at least 24GB VRAM to use GPU and
                        64GB to use CPU but it will likely be very slow. I have not tested this.
  --max_new_tokens MAX_NEW_TOKENS
                        In theory this should change length of generated captions, but it doesn't seem to work
                        currently. (default: 48)
  --find [FIND]         A txt file containing one entry per line of strings to replace in the BLIP caption. Only works
                        if --replace is also specific, ignored otherwise. (default: data/female.txt)
  --replace [REPLACE]   A string that will be used to replace the entries specific in the --find file, ex. 'john doe'
                        (default: None)
  --yaml                Write yaml files instead of txt. Recomended if your trainer supports it for flexibility later.
  --mediums [MEDIUMS]   A text file with list of mediums/photo styles for tags. (default: 'data/mediums.txt')
  --emotions [EMOTIONS]
                        A text file with list of emotions/facial expressions for tags. (default: 'data/emotions.txt')
```

The simplest usage would be to run the following command:
```
python caption.py --replace "jane doe"
```
This will scan all files in the ```output``` folder and all subfolders (i know its weird to default to output here, but its meant to be used after the crop script and that's where that puts its output by default). For each image a txt file with the same name will be created with the BLIP2 caption and CLIP tags as described above.

NOTE: the find and replace functionality will likely require some tuning on your part as the BLIP captions sometimes feature new adjectives (expecially nationalities and such). If that happens you just need to add new strings to appropriate text file (ex: 'data\female.txt') for the terms its missing. For example if you got a caption with 'a young Polish woman in a park' it won't currently replace 'a young Polish woman' because its not in the female.txt. Just add it as a new line and rerun the script and it will replace it. Its impossible to account for everything here so it has to be left for the end user to fine-tune the txt files based on their dataset.
