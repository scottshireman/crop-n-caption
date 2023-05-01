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
import re
import unidecode


def main(args):

    for directory in next(os.walk(args.img_dir))[1]:

        split_dir = directory.split(' and ')

        multiple_names = []
        for split_name in split_dir:

            hanzi_regex = '\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A'
            japanese_hiragana_regex = '\u3041-\u3096'
            japanese_katakana_regex = '\u30A0-\u30FF'

            japanese_regex_string = rf"[{hanzi_regex}{japanese_hiragana_regex}{japanese_katakana_regex}]*"
            regex_string = r"[a-zA-Z\s]*"

            # Look for folders that start with English string
            match_name = re.match(regex_string,split_name)

            if match_name:
                match_name = match_name[0].strip()
                old_path = os.path.join(args.img_dir, directory)
                new_path = os.path.join(args.img_dir, match_name)
                if os.path.exists(os.path.join(args.img_dir, match_name)):
                    if old_path != new_path:
                        for root, dirs, files in os.walk(old_path):
                            for file in files:
                                if os.path.exists(os.path.join(new_path,file)):
                                    file_split = file.split('.')
                                    file_split[0] = file_split[0] + 'b'
                                    new_file = '.'.join(file_split)
                                    shutil.move(os.path.join(root,file), os.path.join(new_path,new_file))

                                else:   
                                    shutil.move(os.path.join(root,file), new_path)

                            try:
                                os.rmdir(old_path)
                            except:
                                print(f"Couldn't remove {old_path}")

                else:
                    os.rename(old_path, new_path)

            else:
            # Look for folders that start with Japanese string
                match_name = re.fullmatch(japanese_regex_string,split_name)

                if match_name:
                    match_name = match_name[0].strip()
                    print(directory)
                    print(unidecode(match_name))



            
    

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
