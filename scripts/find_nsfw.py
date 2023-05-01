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

SUPPORTED_EXT = [".webp", ".jpg", ".png", ".jpeg", ".bmp", ".jfif"]

def check_for_nsfw(folder_path):

    # os.walk all files in folder_path recursively
    for root, dirs, files in os.walk(folder_path, topdown=True):
        [dirs.remove(d) for d in list(dirs) if d in ['face','check']]
        for file in files:
            file_split = os.path.splitext(file)
            if file_split[1] == '.yaml':
                
                full_file_path = os.path.join(root, file)
                for image_extension in SUPPORTED_EXT:
                    full_image_path = os.path.join(root, file_split[0] + image_extension)
                    if os.path.exists(full_image_path):
                        found_image_file = True
                        break  

                if not found_image_file:
                    print(f"Couldn't find image file corresponding to {file}.")
                    continue

                with open(full_file_path, 'r') as f :
                    caption_text = f.read()
                f.close()

                if "a naked photo" in caption_text:
                    return True

    return False



    multiplier = 1
    if person_images > face_images:
        multiplier = face_images / person_images

    effective_images = face_images + multiplier * person_images

    multiply_file = os.path.join(folder_path,'multiply.txt')
    if args.multiply:
        with open(multiply_file, 'w') as f :
            f.write(str(multiplier))

        f.close()
        print(f"{face_images} face images and {person_images} person images.")
        print(f"Created multiply file with {multiplier} for {effective_images} effective images.")
        print(f"Also moved {check_images} images to check folder and/or deleted {deleted_images} images.")
    else:
        print(f"{face_images} face images and {person_images} person images.")

def main(args):

    if args.folders:
        for file_object in os.scandir(args.img_dir):
            if file_object.is_dir():
                if check_for_nsfw(file_object.path):
                    shutil.move(file_object.path, args.nsfw_dir)
                else:
                    shutil.move(file_object.path, args.sfw_dir)

    else:
        if check_for_nsfw(args.img_dir):
            shutil.move(file_object.path, args.nsfw_dir)
        else:
            shutil.move(file_object.path, args.sfw_dir)



    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--nsfw_dir", type=str, default="input", help="Path to put NSFW folders")
    parser.add_argument("--sfw_dir", type=str, default="input", help="Path to put SFW folders")
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Move images with captions that don't have the name of the folder to a folder. Default deletes them."
        )
    parser.add_argument(
        "--multiply",
        action="store_true",
        default=False,
        help="Create a multiply.txt file to set ratio of face to non-face images at 1:1."
        )
    parser.add_argument(
        "--folders",
        action="store_true",
        default=False,
        help="img_dir is a folder of folders to be done seperately (as opposed to one folder with images)."
        )


    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
