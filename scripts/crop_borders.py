"""
Copyright [2023] Scott Shireman

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import argparse
import time
import cv2
import numpy
import requests
import shutil
import math

from PIL import Image
from PIL import ImageOps
from PIL import ImageFile
from tqdm import tqdm
from skimage.metrics import structural_similarity


#person and face recognition
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import face_detection


SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]
VIDEO_EXT = [".mp4", ".mov"]

def get_args(**parser_kwargs):
    """Get command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="input",
        help="Path to input images. (default: 'input')"
        )
    parser.add_argument(
        "--img_done",
        type=str,
        default=None,
        help="If specified, moves processed files to this directory after they've been processed"
        )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Path to output folder for cropped images. (default: 'output')"
        )
    parser.add_argument(
        "--folders",
        action="store_true",
        default=False,
        help="Recurse through each subfolder in img_dir and run the script at that level instead of top level."
        )
    parser.add_argument(
        "--crop_people",
        type=int,
        default=None,
        help="Crop images of people if the probability is greater than the value specified value specified (example: 50)"
        )
    parser.add_argument(
        "--crop_faces",
        type=int,
        default=None,
        help="Crop images of faces if the probability is greater than the value specified value specified (example: 50)"
        )
    parser.add_argument(
        "--skip_multiples",
        action="store_true",
        default=False,
        help="Some images have lots of people/faces and by default all will be extracted. For very large datasets that might mean a lot of false negatives. Set this option to ignore any input image that has multiple people or faces (evaluated seperately)."
        )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        default=False,
        help="By default cropped output images will be written as compressed webp files. Use this flag to instead save them in the same format as the original input image.",
    )
    parser.add_argument(
        "--max_mp",
        type=float,
        default=1.5,
        help="Maximum megapixels to save cropped output images. Larger images will be shrunk to this value. Images with not be resized at all if --no_compress flag is set. (default: 1.5)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Webp quality level to save cropped output images. Will not apply if --no_compress flag is set. (default: 95)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files in output directory if they already exist",
    )
    parser.add_argument(
        "--no_transpose",
        action="store_true",
        default=False,
        help="By default images will be transposed to proper orientation if exif data on the roper orientation exists even if --no_compress is specified. Set this flag to disable.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=0,
        help="Also extract every n frame of videos found where is the integer specified. (default: 0 = won't extract videos)",
    )
    parser.add_argument(
        "--blur_threshold",
        type=int,
        default=0,
        help="If extracting from video specify blur threshold for filtering out blurry images. 100 is a good value. (default: 0 = will keep all)",
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=512,
        help="The resolution at which you intend to train. Cropped images that are smaller than that will be written to 'small' subfolder created in the output folder. Specify 0 to ignore. (default: 512)",
    )
    
    args = parser.parse_args()
    args.max_mp = args.max_mp * 1024000
    args.training_size = args.training_size * args.training_size
    return args

def open_image(full_file_path):

    open_cv_image = cv2.imread(full_file_path)
    
##    ImageFile.LOAD_TRUNCATED_IMAGES = True
##    #Open with PIL and convert to opencv because PIL handles special characters in file names and opencv does not
##    pil_image = Image.open(full_file_path)
##    pil_image = transpose(pil_image)
##    open_cv_image = numpy.array(pil_image)
##    pil_image.close()
##    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    return open_cv_image


def save_image(open_cv_image, full_file_path, args):

    if full_file_path:
        if not os.path.exists(full_file_path) or args.overwrite:
            if not args.no_compress:
                return cv2.imwrite(full_file_path, open_cv_image, [int(cv2.IMWRITE_WEBP_QUALITY), args.quality])

            else:
                return cv2.imwrite(full_file_path, open_cv_image)

    return False

def process_folder(folder_path, person_processor, person_model, face_detector, device, args):


    folder_names = ""

    # os.walk all files in folder_path recursively
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, position=0, leave=True, desc=os.path.basename(folder_path)):
        # for file in files:

            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            full_file_path = os.path.join(root, file)

            if ext.lower() in SUPPORTED_EXT:
                image = open_image(full_file_path)
                save_image(crop_borders(image), full_file_path, args)

def crop_borders(image):

    width = image.shape[1]
    height = image.shape[0]

    top = 150
    bottom = height - 150
    left = 0
    right = width

    image = image[top:bottom, left:right]
    
    return image
    
def main():

    start_time = time.time()
    files_processed = 0
    people_extracted = 0
    faces_extracted = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    
    # os.walk all files in folder_path recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in tqdm(files, position=0, leave=True, desc=os.path.basename(root)):

            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            full_file_path = os.path.join(root, file)

            if ext.lower() in SUPPORTED_EXT:
                image = open_image(full_file_path)
                save_image(crop_borders(image), full_file_path, args)      
                      


if __name__ == "__main__":
    main()
