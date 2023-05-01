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
from tqdm import tqdm


SUPPORTED_EXT = [".webp", ".jpg", ".png", ".jpeg", ".bmp", ".jfif"]

def copy_image_caption(folder_path, destination_path, search_string):

    # os.walk all files in folder_path recursively
    for root, dirs, files in os.walk(folder_path, topdown=True):
        [dirs.remove(d) for d in list(dirs) if d in ['face','check']]
        for file in tqdm(files):
            file_split = os.path.splitext(file)
            if file_split[1] == '.yaml':
                
                full_file_path = os.path.join(root, file)
                for image_extension in SUPPORTED_EXT:
                    full_image_path = os.path.join(root, file_split[0] + image_extension)
                    if os.path.exists(full_image_path):
                        found_image_file = True
                        break  

                if not found_image_file:
                    continue

                with open(full_file_path, 'r') as f :
                    caption_text = f.read()
                f.close()

                if search_string in caption_text:
                    if os.path.exists(full_image_path):
                        shutil.copy(full_file_path, destination_path)
                        shutil.copy(full_image_path, destination_path)


def main(args):


    copy_image_caption(args.img_dir, args.out_dir, args.caption)


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="input", help="Path to put images with the outfit caption")
    parser.add_argument("--caption", type=str, default="input", help="Outfit string to look for")



    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
