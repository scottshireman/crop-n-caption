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

                file_name = file

                while file_name[0] == "_":
                    file_name = file_name[1:]

                print (file_name)
                    
                folder_name = file_name.split("__")[0]
                new_file_name = "__".join(file_name[1:])
                    
                folder_path = os.path.join(args.out_dir,folder_name)


                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                shutil.move(os.path.join(root,file), folder_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")
    parser.add_argument("--min_size", type=int, default=512, help="Size you intend to train at")



    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
