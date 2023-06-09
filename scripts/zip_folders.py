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
import zipfile


# Create a new zip file
def create_new_zip_file(current_zip_number, zip_name, file_type, prefix_folder):
     
    zip_filename = f"{zip_name}_{file_type}_{prefix_folder}{current_zip_number:03d}.zip"
    print(f" Creating {zip_filename}")
    zip_file_path = os.path.join(args.out_dir, zip_filename)
    current_zip_number += 1
    
    return zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED), current_zip_number


def main(args):

    image_extensions = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]
    caption_extensions = [".txt", ".yaml"]

    if args.images_only:
        extensions_to_zip = image_extensions
        file_type = "images"
    elif args.captions_only:
        extensions_to_zip = caption_extensions
        file_type = "captions"
    else:
        extensions_to_zip = image_extensions + caption_extensions
        file_type = "all"

        
    for top_subfolder in os.listdir(args.img_dir):
        top_subfolder_path = os.path.join(args.img_dir, top_subfolder)
        if os.path.isdir(top_subfolder_path):

            # Initialize current zip file number

            current_zip_number = 1
            # Create the first zip file
            zip_file, current_zip_number = create_new_zip_file(current_zip_number, top_subfolder, file_type, args.prefix_folder)

            # Loop through all files in the directory, including subdirectories
            for root, dir, files in os.walk(top_subfolder_path):
                for filename in files:
                    #get file extension
                    file_name_and_extension = os.path.splitext(filename)
                    file_name = file_name_and_extension[0]
                    ext = file_name_and_extension[1]
                    if ext.lower() in extensions_to_zip:
                    
                        file_path = os.path.join(root, filename)
                        
                        # Check if adding the file will exceed the maximum zip file size
                        file_size = os.path.getsize(file_path)
                        if zip_file.infolist() and current_zip_file_size(zip_file) + file_size > args.max_size * 1024 * 1024:
                            # Close the current zip file and create a new one
                            zip_file.close()
                            zip_file, current_zip_number = create_new_zip_file(current_zip_number, top_subfolder, file_type, args.prefix_folder)

                        # Add the file to the current zip file with the preserved folder structure
                        relative_path = os.path.join(args.prefix_folder,os.path.relpath(file_path, args.img_dir))
                        zip_file.write(file_path, relative_path)

            # Close the last zip file
            zip_file.close()

    print("Compression complete!")



# Helper function to calculate the current size of a zip file
def current_zip_file_size(zip_file):
    return sum(fileinfo.file_size for fileinfo in zip_file.infolist())

# Example usage: compress_images_to_zip("/path/to/images/directory")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")
    parser.add_argument("--prefix_folder", type=str, default="training", help="Folder name to be prefixed to the relative path of the files in the zip")
    parser.add_argument("--max_size", type=int, default=512, help="max size of zip files in MB")
    parser.add_argument(
        "--images_only",
        action="store_true",
        default=False,
        help="Only zip image files."
        )
    parser.add_argument(
        "--captions_only",
        action="store_true",
        default=False,
        help="Only zip caption files (txt and yaml)."
        )

    args = parser.parse_args()

    print(f"** Compressing folders snd files in: {args.img_dir}")
    main(args)
