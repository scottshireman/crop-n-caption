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

    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in SUPPORTED_EXT:

                while file_name[0] == "_":
                    file_name = file_name[1:]

                print (f"processing {file_name} . . . ")

                file_name_parts = file_name.split("_")
                    
                if len(file_name_parts) > 2:
                    new_folder_name = file_name_parts[0] + " " + file_name_parts[1]
                    new_file_name = "__".join(file_name_parts[2:]) + ext.lower()
                else:
                    print(f"{file_name} doesn't meet expectations")
                    
                    
                folder_path = os.path.join(args.out_dir,new_folder_name)
                new_file_path = os.path.join(folder_path,new_file_name)

                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                shutil.copy(os.path.join(root,file), new_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")


    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
