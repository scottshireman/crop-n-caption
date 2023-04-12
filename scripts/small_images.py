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
import shutil
import cv2
import mediapipe as mp
import numpy

from imageai.Detection import ObjectDetection
from PIL import Image

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def main(args):

    min_pixels = args.min_size * args.min_size
    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)

                image = Image.open(full_file_path)

                width = image.width
                height = image.height
                pixels = width * height

                print (file + ": " + str(pixels))
                
                if pixels < min_pixels:
                    image.close()
                    shutil.move(full_file_path, args.out_dir + "\\" + file)
                else:
                    image.close()
                    print("no issue")

    print("Minimum pixels: " + str(min_pixels))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")
    parser.add_argument("--min_size", type=int, default=512, help="Size you intend to train at")



    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
