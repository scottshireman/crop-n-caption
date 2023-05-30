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
import subprocess
from tqdm import tqdm

SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]


def upscale_image(input_object, output_folder):

    photoai_cli_path = "C:/Program Files/Topaz Labs LLC/Topaz Photo AI/tpai.exe"  # Windows

    if not os.path.exists(photoai_cli_path):
        print("CLI not found. Check the path.")
        quit()

    command = [
        photoai_cli_path,
        input_object,
        "-output", output_folder
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    
    except subprocess.CalledProcessError as error:
        return False

def process_folder(folder_path, output_path):

    basename = os.path.basename(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, position=0, leave=True, desc=basename):
            
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            full_file_path = os.path.join(root, file)

            if ext.lower() in SUPPORTED_EXT:
                new_file_name = file_name + "_topaz" + ext
                new_file_path = os.path.join(root, new_file_name)

                os.rename(full_file_path, new_file_path)

                if upscale_image(new_file_path, output_path):
                    if os.path.exists(os.path.join(output_path, new_file_name)):
                        os.remove(new_file_path)
                              

def main(args):

    if args.folders:
        for file_object in os.scandir(args.img_dir):
            if file_object.is_dir():
                output_path = os.path.join(args.out_dir, file_object.name)
                process_folder(file_object, output_path)

    else:
        process_folder(file_object.path, args.out_dir)
                
    



# Helper function to calculate the current size of a zip file
def current_zip_file_size(zip_file):
    return sum(fileinfo.file_size for fileinfo in zip_file.infolist())

# Example usage: compress_images_to_zip("/path/to/images/directory")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")
    parser.add_argument(
        "--folders",
        action="store_true",
        default=False,
        help="Recurse through each subfolder in img_dir and run the script at that level instead of top level."
        )


    args = parser.parse_args()

    main(args)
