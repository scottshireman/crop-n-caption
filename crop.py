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
import cv2
import numpy
import requests
import shutil
import math
import dlib

from PIL import Image
from PIL import ImageOps
from PIL import ImageFile
from tqdm import tqdm

#person and face recognition
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import face_detection


SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]
VIDEO_EXT = [".mp4", ".mov"]

def get_args(**parser_kwargs):
    """Get command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="input",
        help="Path to input images. (default: 'input')"
        )
    parser.add_argument(
        "--img_done",
        type=str,
        default=None,
        help="If specified, moves processed files to this directory after they've been processed"
        )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Path to output folder for cropped images. (default: 'output')"
        )
    parser.add_argument(
        "--append_folder_name",
        action="store_true",
        default=False,
        help="Prepends to output filenames the names of all subfolders the input file is nested in, delimitted by '_' Helpful for some large datasets with lots of meaningful subfolders that can be used later for caption tags."
        )
    parser.add_argument(
        "--crop_people",
        type=int,
        default=None,
        help="Crop images of people if the probability is greater than the value specified value specified (example: 50)"
        )
    parser.add_argument(
        "--crop_faces",
        type=int,
        default=None,
        help="Crop images of faces if the probability is greater than the value specified value specified (example: 50)"
        )
    parser.add_argument(
        "--skip_multiples",
        action="store_true",
        default=False,
        help="Some images have lots of people/faces and by default all will be extracted. For very large datasets that might mean a lot of false negatives. Set this option to ignore any input image that has multiple people or faces (evaluated seperately)."
        )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        default=False,
        help="By default cropped output images will be written as compressed webp files. Use this flag to instead save them in the same format as the original input image.",
    )
    parser.add_argument(
        "--max_mp",
        type=float,
        default=1.5,
        help="Maximum megapixels to save cropped output images. Larger images will be shrunk to this value. Images with not be resized at all if --no_compress flag is set. (default: 1.5)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Webp quality level to save cropped output images. Will not apply if --no_compress flag is set. (default: 95)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files in output directory if they already exist",
    )
    parser.add_argument(
        "--no_transpose",
        action="store_true",
        default=False,
        help="By default images will be transposed to proper orientation if exif data on the roper orientation exists even if --no_compress is specified. Set this flag to disable.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=0,
        help="Also extract every n frame of videos found where is the integer specified. (default: 0 = won't extract videos)",
    )
    parser.add_argument(
        "--blur_threshold",
        type=int,
        default=0,
        help="If extracting from video specify blur threshold for filtering out blurry images. 100 is a good value. (default: 0 = will keep all)",
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=512,
        help="The resolution at which you intend to train. Cropped images that are smaller than that will be written to 'small' subfolder created in the output folder. Specify 0 to ignore. (default: 512)",
    )
    
    args = parser.parse_args()
    args.max_mp = args.max_mp * 1024000
    args.training_size = args.training_size * args.training_size
    return args

def open_image(full_file_path):

    open_cv_image = cv2.imread(full_file_path)
    
##    ImageFile.LOAD_TRUNCATED_IMAGES = True
##    #Open with PIL and convert to opencv because PIL handles special characters in file names and opencv does not
##    pil_image = Image.open(full_file_path)
##    pil_image = transpose(pil_image)
##    open_cv_image = numpy.array(pil_image)
##    pil_image.close()
##    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    return open_cv_image

def get_write_path(open_cv_image, filename, extension, args):

    full_file_path = None
    dimensions = open_cv_image.shape[0] * open_cv_image.shape[1]

    if dimensions < args.training_size / 4:
        return None

    else:
        if not args.no_compress:
            if dimensions > args.training_size:
                full_file_path = os.path.join(args.out_dir,filename+'.webp')

            else:
                full_file_path = os.path.join(args.out_dir,"small",filename+'.webp')

        else:
            if dimensions > args.training_size:
                full_file_path = os.path.join(args.out_dir,filename+extension)
            else:
                full_file_path = os.path.join(args.out_dir,"small",filename+extension)

        return full_file_path
        

def save_image(open_cv_image, full_file_path, args):

    if full_file_path:
        if not os.path.exists(full_file_path) or args.overwrite:
            if not args.no_compress:
                return cv2.imwrite(full_file_path, open_cv_image, [int(cv2.IMWRITE_WEBP_QUALITY), args.quality])

            else:
                return cv2.imwrite(full_file_path, open_cv_image)

    return False
                    

def check_shape(box_points, image_width, image_height):
    #ED2 requires rations between 1:4 and 4:1 so we need to adjust the box points accordingily

    # convert box points to integers
    person_width = box_points[2]-box_points[0]
    person_height = box_points[3]-box_points[1]

    if person_height > person_width * 4:
        new_width = (person_height // 4)
        box_points[0] = box_points[0] - (new_width - person_width) // 2 - 1
        box_points[2] = box_points[2] + (new_width - person_width) // 2 + 1

        if box_points[0] < 0:
            box_points[2] = box_points[2] - box_points[0]
            box_points[0] = 0

        if box_points[2] > image_width:
            box_points[0] = box_points[0] + image_width - box_points[2]
            box_points[2] = image_width

            if box_points[0] < 0:
                box_points[0] = 0

    if  person_width > person_height * 4:
        new_height = (person_width // 4)
        box_points[1] = box_points[1] - (new_height - person_height) // 2 - 1
        box_points[3] = box_points[3] + (new_height - person_height) // 2 + 1

        if box_points[1] < 0:
            box_points[3] = box_points[3] - box_points[1]
            box_points[1] = 0

        if box_points[3] > image_width:
            box_points[1] = box_points[1] + image_width - box_points[3]
            box_points[3] = image_width

            if box_points[1] < 0:
                box_points[1] = 0

    return box_points
    

def get_inflated_face_image(image, top, bottom, left, right, inflation):

    width = right - left
    height = bottom - top
    
    width_inflation, height_inflation = width * inflation, height * inflation

    return crop_within_bounds(
        image,
        round(top - height_inflation),      # top
        round(bottom + height_inflation),   # bottom
        round(left - width_inflation),      # left
        round(right + width_inflation)      # right
    )

def crop_within_bounds(image, top, bottom, left, right):

    if top < 0: top = 0
    elif top >= image.shape[0]: top = image.shape[0] - 1

    if bottom < 0: bottom = 0
    elif bottom >= image.shape[0]: bottom = image.shape[0] - 1

    if left < 0: left = 0
    elif left >= image.shape[1]: left = image.shape[1] - 1

    if right < 0: right = 0
    elif right >= image.shape[1]: right = image.shape[1] - 1

    return image[top:bottom+1, left:right+1]


def check_blur(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return int(cv2.Laplacian(gray_image, cv2.CV_64F).var())

def check_frame_diff(image_new, image_old):

    width = 1024
    height = 1024
    dim = (width, height)

    #convert to grayscale and same dimensions for simple comparison.
    image_new = cv2.cvtColor(cv2.resize(image_new, dim, interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    image_old = cv2.cvtColor(cv2.resize(image_old, dim, interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(image_new, image_old).sum() / 1024 / 1024 / 256

    return diff

def crop_away_blackbars(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)

    image = image[y:y+h, x:x+w]

    return image


def extract_from_video(full_file_path, face_detector, person_model, person_processor, args, device):

    # For each frame in a video
    # Find faces and people
    # If they pass a basic blurryness threshold
    # See if they're different than the last face or person
    # If they are, find the least blurry image from the last block and add it to keyframes
    # If not, add them to the current block and move on
    # Return keyframes
    
    unique_face_images = []
    unique_face_encodings = []
    
    person_images = []
    unique_person_images = []
    unique_person_encodings = []

    first_face = True
    first_person = True

    video = cv2.VideoCapture(full_file_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in tqdm(range(frame_count), desc='Extracting faces and people from frames', position=1, leave=False):
    
        video.set(1, frame_id);
        success, image = video.read()

        if success:
            frame_face_images, frame_face_encodings = get_face_images(image, face_detector)
            frame_person_images, frame_person_encodings = get_person_images(image, args.crop_people, person_model, person_processor, device)

            if len(frame_face_images) == 1:
                if check_blur(frame_face_images[0]) > args.blur_threshold:
                    if first_face:
                        # First face always starts a new block of similar faces.
                        face_images = []
                        face_encodings = []
                        face_images.append(frame_face_images[0])
##                        face_encodings.append(frame_face_encodings[0])
                        last_face = frame_face_images[0]
                    else:
                        if check_frame_diff( frame_face_images[0], last_face ) > threshold:
                            # Found a signficant change, ie, the start of a new block of similar faces!
                            # Get the least blurry face from the previous block.
                            least_blurry_frame_id = find_least_blurry_frame_id(face_images)

                            unique_face_images.append(face_images[least_blurry_frame_id])
##                            unique_face_encodings.append(face_encodings[least_blurry_frame_id])
                            

                            # Then, start a new block of similar faces with the current face.
                            face_images = []
                            face_encodings = []
                            face_images.append(frame_face_images[0])
##                            face_encodings.append(frame_face_encodings[0])
                            last_face = frame_face_images[0]

                        else:
                            # Didn't find start of a new block so just append to old block and move on.
                            face_images.append(frame_face_images[0])
##                            face_encodings.append(frame_face_encodings[0])
                            last_face = frame_face_images[0]

            if len(frame_person_images) == 1:
                if check_blur(frame_person_images[0]) > args.blur_threshold:
                    if first_person:
                        # First face always starts a new block of similar faces.
                        person_images = []
                        person_images.append(frame_person_images[0])
##                        person_encodings.append(frame_person_encodings[0])
                        last_person = frame_person_images[0]
                    else:
                        if check_frame_diff( frame_person_images[0], last_person ) > threshold:
                            # Found a signficant change, ie, the start of a new block of similar faces!
                            # Get the least blurry face from the previous block.
                            least_blurry_frame_id = find_least_blurry_frame_id(person_images)
                            unique_person_images.append(person_images[least_blurry_frame_id])
##                            unique_person_encodings.append(person_encodings[least_blurry_frame_id])


                            # Then, start a new block of similar faces with the current face.
                            person_images = []
                            person_encodings = []

                            person_images.append(frame_person_images[0])
##                            person_encodings.append(frame_person_encodings[0])
                            last_person = frame_person_images[0]

                        else:
                            # Didn't find start of a new block so just append to old block and move on.
                            person_images.append(frame_person_images[0])
##                            person_encodings.append(frame_person_encodings[0])

                            last_person = frame_person_images[0]

    video.release()

    # Get the least blurry face from the last block of similar faces.
    if len(face_images) > 0:
        least_blurry_frame_id = find_least_blurry_frame_id(face_images)
        unique_face_images.append(face_images[least_blurry_frame_id])
##        unique_face_encodings.append(face_encodings[least_blurry_frame_id])
        face_images = []

    # Get the least blurry person from the last block of similar persons.
    if len(person_images) > 0:
        least_blurry_frame_id = find_least_blurry_frame_id(person_images)
        unique_person_images.append(person_images[least_blurry_frame_id])
##        unique_person_encodings.append(person_images[least_blurry_frame_id])

        person_images = []


    return unique_face_images, unique_face_encodings, unique_person_images, unique_person_encodings


def find_least_blurry_frame_id(frames):

    if len(frames) > 0:

        highest_blur_score = 0
        least_blurry_frame = 0

        for frame_id, frame in enumerate(frames):
            blur_score = check_blur(frame)

            if blur_score > highest_blur_score:
                highest_blur_score = blur_score
                least_blurry_frame = frame_id

        return least_blurry_frame


def get_person_images(image, threshold, model, processor, device, object_to_find = 'person'):

    objects_detected = []
    encodings = []
    
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold/100)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        if model.config.id2label[label.item()] == object_to_find:

            box_points = check_shape(box, image.shape[1], image.shape[0])
            
            cropped_image = crop_within_bounds(image, box_points[1], box_points[3], box_points[0], box_points[2])
            objects_detected.append(cropped_image)
##            _, encodings = get_face_images(cropped_image)
##            encodings.append(encodings[0])
            
    return objects_detected, encodings


def get_face_images(image, detector):

    objects_detected = []
    encodings = []

    detections = detector.detect(image)

    for detection in detections:
        box = [round(detection[i]) for i in range(0,4)]
        box_points = check_shape(box, image.shape[1], image.shape[0])
        objects_detected.append(get_inflated_face_image(image, box_points[1], box_points[3], box_points[0], box_points[2], 0.4))

    return objects_detected, encodings

def encoding_stuff():

    return True
    
def main():

    start_time = time.time()
    files_processed = 0
    people_extracted = 0
    faces_extracted = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    
    #Load person detection model
    person_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    person_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    person_model.to(device)

    #Load face detection model
    # face_threshold = args.crop_faces / 100 
    face_detector = face_detection.build_detector("DSFDDetector", confidence_threshold=args.crop_faces/100, nms_iou_threshold=.3, device=device)
    
    folder_names = ""

    small_dir = "small"
    if args.training_size > 0 and not os.path.exists(os.path.join(args.out_dir,small_dir)):
        os.mkdir(os.path.join(args.out_dir,small_dir))

    print(f"\nExtracting files in {args.img_dir}")
    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        tqdm_files = tqdm(files, position=0, leave=True)
        for file in tqdm_files:
            tqdm_files.set_description(f"Extracting from {file}")
            tqdm_files.refresh()
            
            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            full_file_path = os.path.join(root, file)

            if ext.lower() in SUPPORTED_EXT:
                folder_names = root.replace(args.img_dir,"")
                relative_path = os.path.join(folder_names, file)
                folder_names = folder_names.replace("\\","_")
                
                if args.append_folder_name:
                    
                    #PIL allowed us to open files with special characters, but we need to remove them out before writing
                    if folder_names.isalnum() is False:
                        clean_string = "".join(ch for ch in folder_names if (ch.isalnum() or ch == " " or ch == "_"))
                        folder_names = clean_string

                    if folder_names != "":
                        if folder_names[0] == "_":
                            folder_names = folder_names[1:]

                    if folder_names != "":
                        folder_names = folder_names + "__"
                        
                else:
                    folder_names = ""

                image = open_image(full_file_path)
                
                people_extracted_from_image, person_encodings = get_person_images(image, args.crop_people, person_model, person_processor, device)
                if len(people_extracted_from_image) > 0:
                    for person_no, person in enumerate(people_extracted_from_image):
                        full_write_path = get_write_path(person, file_name + "_person_" + str(person_no).zfill(3), ext, args)
                        if save_image(person, full_write_path, args):
                            #do encoding stuff
                            encoding_stuff()
                            

                people_extracted_from_image = []

                faces_extracted_from_image, face_encodings = get_face_images(image, face_detector)
                if len(faces_extracted_from_image) > 0:
                    for face_no, face in enumerate(faces_extracted_from_image):
                        full_write_path = get_write_path(face, file_name + "_face_" + str(face_no).zfill(3), ext, args)
                        if save_image(face, full_write_path, args):
                            #do encoding stuff
                            encoding_stuff()
                        
                faces_extracted_from_image = []


            elif args.video != 0 and ext.lower() in VIDEO_EXT:

                faces_extracted_from_image, face_encodings, people_extracted_from_image, person_encodings = \
                            extract_from_video(full_file_path, face_detector, person_model, person_processor, args, device)

                if len(people_extracted_from_image) > 0:
                    for person_no, person in enumerate(people_extracted_from_image):
                        full_write_path = get_write_path(person, file_name + "_person_" + str(person_no).zfill(3), ext, args)
                        if save_image(person, full_write_path, args):
                            #do encoding stuff
                            encoding_stuff()

                people_extracted_from_image = []

                if len(faces_extracted_from_image) > 0:
                    for face_no, face in enumerate(faces_extracted_from_image):
                        full_write_path = get_write_path(face, file_name + "_face_" + str(face_no).zfill(3), ext, args)
                        if save_image(face, full_write_path, args):
                            #do encoding stuff
                            encoding_stuff()
                            
                faces_extracted_from_image = []

            if args.img_done is not None:
                    if os.path.exists(args.img_done) and not os.path.exists(os.path.join(args.img_done,file)):
                        shutil.move(full_file_path, args.img_done)
                        
                

    exec_time = time.time() - start_time
    print(f"  Processed {files_processed} files in {exec_time} seconds for an average of {exec_time/files_processed} sec/file.")
    print(f"  Extracted {people_extracted} images of people and {faces_extracted} images of faces.")        


if __name__ == "__main__":
    main()
