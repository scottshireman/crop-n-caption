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

def keep_first(folder_path, args):

    first_file = True
    # os.walk all files in folder_path recursively
    for root, dirs, files in os.walk(folder_path, topdown=True):

        for file in tqdm(files):

            if 'unique' in root:
                shutil.move(os.path.join(root,file), args.img_dir)
            
            elif first_file or not args.first_only:
                shutil.move(os.path.join(root,file), args.img_dir)
                first_file = False
                
            else:
                os.remove(os.path.join(root,file))

    os.rmdir(folder_path)


def main(args):

    for directory in tqdm(next(os.walk(args.img_dir))[1]):
        keep_first(os.path.join(args.img_dir,directory), args)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument(
        "--first_only",
        action="store_true",
        default=False,
        help="Move images with captions that don't have the name of the folder to a folder. Default deletes them."
        )



    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
