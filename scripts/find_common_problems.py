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
from unidecode import unidecode

SUPPORTED_EXT = [".yaml"]

def find_bad_yaml(full_file_path):
    #print(full_file_path)
    with open(full_file_path, 'r') as file :
        lines = file.readlines()

        for line in lines:
            if line.rstrip() == '  - tag:':
                print(f"Empty tag: {full_file_path}")

            split_line = line.split(':')
            if len(split_line) > 1:
                if split_line[1].strip() == '' and split_line[0].strip() != 'tags':
                    print(f"Bad character in {split_line[1]} in {full_file_path}")

    with open(full_file_path, 'r') as file :
        file_text = file.read()

    if file_text != unidecode(file_text):
        print(f"non-ascii character found in {full_file_path}")
        file_text = unidecode(file_text)

        with open(full_file_path, 'w') as file :
            file.write(file_text)
        



def main(args):


    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:

            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root,file)
                find_bad_yaml(full_file_path)

                image_file = file_name_and_extension[0]+'.webp'
                if not os.path.exists(os.path.join(root,image_file)):
                    print(f"Missing image file: {image_file}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")



    args = parser.parse_args()

    main(args)
