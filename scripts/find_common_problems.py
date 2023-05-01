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
    print(f"Fixing {full_file_path} . . .")

    write_file = False
    with open(full_file_path, 'r') as file :
        lines = file.readlines()
        new_lines = []

        for line in lines:

            split_line = line.split(':')
            new_line = line
            
            if len(split_line) == 1:
                if split_line[0] != '  - tag:':
                    new_lines.append(new_line)

            else:
                tag_content = ''.join(split_line[1:]).strip()

                if not tag_content.isalnum():
                    #special characters in string
                    allowed_characters = ' +-/'
                    tag_content = unidecode(tag_content)
                    tag_content = "".join(ch for ch in tag_content if ch.isalnum() or ch in allowed_characters)

                if not tag_content.isdigit():
                    new_line = f"{split_line[0]}: {tag_content.strip()}\n"
                    new_lines.append(new_line)

        file.close()

        new_text = ''.join(new_lines)

        with open(full_file_path, 'w') as file :
            file.write(new_text)

        file.close()
        



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

                if not os.path.exists(os.path.join(root,file_name_and_extension[0]+'.webp')) \
                   and not os.path.exists(os.path.join(root,file_name_and_extension[0]+'.jpg')):
                    print(f"Missing image file")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")



    args = parser.parse_args()

    main(args)
