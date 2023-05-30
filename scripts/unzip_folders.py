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
def create_new_zip_file(current_zip_number, zip_name, file_type):
     
    zip_filename = f"{zip_name}_{file_type}_{current_zip_number:03d}.zip"
    print(f" Creating {zip_filename}")
    zip_file_path = os.path.join(args.out_dir, zip_filename)
    current_zip_number += 1
    
    return zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED), current_zip_number

def unzip( zip_file_path, folder_path ):
    
    zip_file = zipfile.ZipFile(zip_file_path)
    
    if zip_file.testzip() is None:

        zip_file.extractall( folder_path )
        zip_file.close()
        return True
    else:
        return False
    

def main(args):

    zip_extensions = [".zip"]

    # Loop through all files in the directory, including subdirectories
    for root, dir, files in os.walk(args.img_dir):
        for filename in files:
            #get file extension
            file_name_and_extension = os.path.splitext(filename)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in zip_extensions:
            
                file_path = os.path.join(root, filename)

                unzip(file_path, args.out_dir)
                



    print("Unzip complete!")



# Helper function to calculate the current size of a zip file
def current_zip_file_size(zip_file):
    return sum(fileinfo.file_size for fileinfo in zip_file.infolist())

# Example usage: compress_images_to_zip("/path/to/images/directory")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--out_dir", type=str, default="output", help="Path to folder for extracted images")


    args = parser.parse_args()

    print(f"** Compressing folders snd files in: {args.img_dir}")
    main(args)
