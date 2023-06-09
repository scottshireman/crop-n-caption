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
import boto3
from tqdm import tqdm

SUPPORTED_EXT = [".zip"]

def download_all_files (bucket, folder_path):

    for item in bucket.objects.all():
        print(f"Downloading {item.key}")
        if folder_path[-1] != '/':
            folder_path = folder_path + '/'
        bucket.download_file(item.key, folder_path+item.key)

def upload_file(bucket, file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    file_size = os.stat(file_name).st_size
    basename = os.path.basename(file_name)

    with tqdm(total=file_size, unit="B", unit_scale=True, desc=basename, position=0, leave=True) as pbar:
        bucket.upload_file(
            file_name,
            object_name,
            ExtraArgs={'ContentType': 'application/zip'},
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )

    #bucket.upload_file(file_name, object_name, ExtraArgs={'ContentType': 'application/zip'})

    return True


def main(args):

    if args.access_file is not None:
        if os.path.isfile(args.access_file):
            with open(args.access_file, 'r') as file :
                lines = file.readlines()
                if len(lines) != 3:
                    print("If specifying an access key file, it must be exactly three lines with " + \
                        "the endpoint URL on the first line, the access key on the second line, " + \
                        "and the secret key on the third line")
                    quit()
                    
                else:
                    endpoint_url = lines[0].strip()
                    aws_access_key_id = lines[1].strip()
                    aws_secret_access_key = lines[2].strip()
        else:
            print("Access key file not found")
            quit()

    elif args.endpoint_url is not None and args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        endpoint_url = args.endpoint_url
        aws_access_key_id = args.aws_access_key_id
        aws_secret_access_key = args.aws_secret_access_key

    else:
        print("You must specify the path to text file with the access info or provide it with the three flags")
        quit()
          

    s3 = boto3.resource('s3',
          endpoint_url = endpoint_url,
          aws_access_key_id = aws_access_key_id,
          aws_secret_access_key = aws_secret_access_key
        )

    bucket = s3.Bucket(args.bucket)
    print(bucket.name)

    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root,file)
                basename = os.path.basename(full_file_path)
                
                if not upload_file(bucket, full_file_path, file):
                    print(f"Error uploading {file}")
                    quit()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--bucket", type=str, default="input", help="Path to images")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Example: https://<application ID>.r2.cloudflarestorage.com")
    parser.add_argument("--aws_access_key_id", type=str, default=None, help="Access key")
    parser.add_argument("--aws_secret_access_key", type=str, default=None, help="Secret key")
    parser.add_argument("--access_file", type=str, default=None, help="A text file with the endpoint URL, access key, and secret key")


    args = parser.parse_args()

    main(args)
