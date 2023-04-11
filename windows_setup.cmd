python -m venv venv
call "venv\Scripts\activate.bat"
echo should be in venv here
cd .
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url "https://download.pytorch.org/whl/cu116"
REM pip install transformers==4.27.1
pip install git+https://github.com/huggingface/transformers.git
pip install diffusers[torch]==0.13.0
pip install pynvml==11.4.1
pip install bitsandbytes==0.35.0
git clone https://github.com/DeXtmL/bitsandbytes-win-prebuilt tmp/bnb_cache
pip install ftfy==6.1.1
pip install aiohttp==3.8.3
pip install tensorboard^>=2.11.0
pip install protobuf==3.20.1
pip install wandb==0.14.0
pip install pyre-extensions==0.0.23
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
::pip install "xformers-0.0.15.dev0+affe4da.d20221212-cp38-cp38-win_amd64.whl" --force-reinstall
pip install pytorch-lightning==1.6.5
pip install OmegaConf==2.2.3
pip install numpy==1.23.5
pip install keyboard
pip install lion-pytorch

pip install cython
pip install pillow^>=7.0.0
pip install opencv-python^>=4.1.2
pip install torchvision^>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install pytest==7.1.3
pip install tqdm==4.64.1
pip install scipy^>=1.7.3
pip install matplotlib^>=3.4.3
pip install mock==4.0.3
pip install imageai --upgrade
pip install mediapipe

pip install torchvision
pip install open_clip_torch
pip install salesforce-lavis


pip install git+https://github.com/pharmapsychotic/clip-interrogator

REM python utils/patch_bnb.py
REM python utils/get_yamls.py
GOTO :eof

:ERROR
echo Something blew up. Make sure Pyton 3.10.x is installed and in your PATH.

:eof
ECHO done
pause
