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
