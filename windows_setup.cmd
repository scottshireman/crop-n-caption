python -m venv venv
call "venv\Scripts\activate.bat"
echo should be in venv here
cd .
python -m pip install --upgrade pip

REM Install the stuff crop.py needs
pip install cython
pip install pillow^>=7.0.0
pip install numpy==1.23.5
pip install opencv-python^>=4.1.2
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url "https://download.pytorch.org/whl/cu116"
pip install torchvision^>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytest==7.1.3
pip install tqdm==4.64.1
pip install scipy^>=1.7.3
pip install matplotlib^>=3.4.3
pip install mock==4.0.3
pip install imageai --upgrade
pip install mediapipe

REM Install additional stuff caption.py needs
pip install git+https://github.com/huggingface/transformers.git
pip install pynvml==11.4.1
pip install git+https://github.com/pharmapsychotic/clip-interrogator

GOTO :eof


:ERROR
echo Something blew up. Make sure Pyton 3.10.x is installed and in your PATH.

:eof
ECHO done
pause
