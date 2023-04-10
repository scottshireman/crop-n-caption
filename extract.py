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
from PIL import Image
from PIL import ImageOps

#person and face recognition
import mediapipe as mp
from imageai.Detection import ObjectDetection


SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_args(**parser_kwargs):
    """Get command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="input",
        help="Path to images"
        )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Path to folder for extracted images"
        )
    parser.add_argument(
        "--person_probability",
        type=int,
        default=10,
        help="Minimum probability"
        )
    parser.add_argument(
        "--face_probability",
        type=int,
        default=10,
        help="Minimum probability"
        )
    parser.add_argument(
        "--append_folder_name",
        action="store_true",
        default=False,
        help="Appends the folder names to the file name"
        )
    parser.add_argument(
        "--extract_people",
        action="store_true",
        default=False,
        help="Extract images of people"
        )
    parser.add_argument(
        "--extract_faces",
        action="store_true",
        default=False,
        help="Extract closeup face images"
        )
    parser.add_argument(
        "--skip_multiples",
        action="store_true",
        default=False,
        help="Don't extract if multiple people or faces exist"
        )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        default=False,
        help="don't shrink large images or convert to webp. saves in original format.",
    )
    
    parser.add_argument(
        "--max_mp",
        type=float,
        default=1.5,
        help="maximum megapixels (default: 1.5)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="save quality (default: 95, range: 0-100, suggested: 90+)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite files in output directory",
    )
    parser.add_argument(
        "--noresize",
        action="store_true",
        default=False,
        help="do not resize, just fix orientation",
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=0,
        help="Size at which you intend to train. Puts smaller files to 'small' subfolder. 0 ignores.",
    )
    
    args = parser.parse_args()
    args.max_mp = args.max_mp * 1024000
    args.training_size = args.training_size * args.training_size
    return args

def open_image(full_file_path):

    #Open with PIL and convert to opencv because PIL handles special characters in file names and opencv does not
    pil_image = Image.open(full_file_path)
    pil_image = transpose(pil_image)
    open_cv_image = numpy.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    return open_cv_image

def save_image(full_file_path, open_cv_image, args):

    #Covert to PIL and write to handle more file types

    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(open_cv_image)

    full_file_path = full_file_path.replace(os.path.splitext(full_file_path)[1], ".webp")

    if args.overwrite or not os.path.exists(full_file_path):  
        pil_image = transpose(pil_image)
            
        if args.no_compress:
            pil_image.save(full_file_path)
        else:
            if oversize(pil_image, args.max_mp):
                pil_image = shrink(pil_image, args)
                                   
            pil_image.save(full_file_path, "webp", quality=args.quality)
                                   

def oversize(img, max_mp):
    """Check if an image is larger than the maximum size."""
    return (img.width * img.height) > max_mp

def transpose(img):
    """Transpose an image."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception as err:
        return img

def shrink(img, args):
    """Shrink an image."""
    hw = img.size
    ratio = args.max_mp / (hw[0]*hw[1])
    newhw = (int(hw[0]*ratio**0.5), int(hw[1]*ratio**0.5))

    try:
        return img.resize(newhw, Image.BICUBIC)
    except Exception as err:
        return img
    
def extract_people(detector, image, args):

    detection_images = []

    custom_objects = detector.CustomObjects(person=True)
        
    detected_image_array, detections = detector.detectObjectsFromImage(
        custom_objects=custom_objects,
        input_image=image,
        output_type="array",
        extract_detected_objects=False,
        minimum_percentage_probability=args.person_probability,
        display_percentage_probability=False,
        display_object_name=False
        )

    if detections is not None:
        
        for detection in detections:

            box_points = check_person_shape(detection["box_points"], image.shape[1], image.shape[0])
            percentage_probability = detection["percentage_probability"]            

            detection_images.append(crop_within_bounds(image, box_points[1], box_points[3], box_points[0], box_points[2]))

    return detection_images

def check_person_shape(box_points, image_width, image_height):
    #ED2 requires rations between 1:4 and 4:1 so we need to adjust the box points accordingily

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
    

def extract_faces(detector, image, args):
    
    inflation_factor = 1.0
    face_images = []
    detected_faces = detector.process(image).detections

    if detected_faces is not None:
        for face in detected_faces:

            # The mp.solutions.face_detection.FaceDetection network may rarely 'find' a face completely outside the image, so ignore those
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:
                inflated_face_image = get_inflated_face_image(image, face.location_data.relative_bounding_box, inflation_factor)

                face_images.append(inflated_face_image)

    return face_images


def get_inflated_face_image(image, face_box, inflation):
    """
    Crop and return the located face in the image. The perimeter of the crop is inflated around the center using the provided inflation factor.
    :param image: The image containing the face.
    :param face_box: The bounding box of the detected face. Must be a mediapipe.framework.formats.location_data_pb2.RelativeBoundingBox object.
    :param inflation: The factor by which the bounding box of a face should be inflated. E.g. 0.5 will inflate the box's perimeter by 50% around the centre.
    :return: A sub-image containing only the face.
    """

    width_inflation, height_inflation = face_box.width * inflation, face_box.height * inflation

    return crop_within_bounds(
        image,
        round((face_box.ymin - height_inflation / 2) * image.shape[0]),                    # top
        round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0]),  # bottom
        round((face_box.xmin - width_inflation / 2) * image.shape[1]),                     # left
        round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1])     # right
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


def main():

    start_time = time.time()
    files_processed = 0
    people_extracted = 0
    faces_extracted = 0
    
    args = get_args()
        

    #Download person detection model if it doesn't already exist in the models subfolder
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("models/retinanet_resnet50_fpn_coco-eeacb38b.pth"):
        r = requests.get('https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/retinanet_resnet50_fpn_coco-eeacb38b.pth', stream=True)
        with open(models/retinanet_resnet50_fpn_coco-eeacb38b.pth, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

    #Load person detection model
    person_detector = ObjectDetection()
    person_detector.setModelTypeAsRetinaNet()
    person_detector.setModelPath("models/retinanet_resnet50_fpn_coco-eeacb38b.pth")
    person_detector.loadModel()

    #Load face detection model
    face_threshold = args.face_probability / 100
    face_detector = mp.solutions.face_detection.FaceDetection( min_detection_confidence = face_threshold, model_selection = 1 )

    folder_names = ""

    small_dir = os.path.join(args.out_dir, "small")
    if args.training_size > 0 and not os.path.exists(small_dir):
        os.mkdir(small_dir)

    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            #get file extension
            file_name_and_extension = os.path.splitext(file)
            file_name = file_name_and_extension[0]
            ext = file_name_and_extension[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                folder_names = root.replace(args.img_dir,"").replace("\\","_")
                relative_path = os.path.join(folder_names, file)
                
                if args.append_folder_name:
                    
                    #PIL allowed us to open files with special characters, but we need to remove them out before writing
                    if folder_names.isalnum() is False:
                        clean_string = "".join(ch for ch in folder_names if (ch.isalnum() or ch == " " or ch == "_"))

                    if folder_names != "":
                        folder_names = folder_names + "__"

                if folder_names[0] == "_":
                    folder_names = folder_names[1:]

                print(f"  Processing {relative_path} . . . ")
                        
                image = open_image(full_file_path)

                if args.extract_people:
                    person_detections = extract_people(person_detector, image, args)

                    if args.skip_multiples is False or len(person_detections) == 1:
                        for person_id, person in enumerate(person_detections):
                            if person.shape[0] * person.shape[1] < args.training_size:
                                save_image(os.path.join(small_dir, folder_names + "person_" + file_name + "_" + str(person_id) + ext), person, args)
                            else:
                                save_image(os.path.join(args.out_dir, folder_names + "person_" + file_name + "_" + str(person_id) + ext), person, args)

                            people_extracted = people_extracted + 1

                if args.extract_faces:
                    face_detections = extract_faces(face_detector, image, args)

                    if args.skip_multiples is False or len(face_detections) == 1:
                        for face_id, face in enumerate(face_detections):
                            if face.shape[0] * face.shape[1] < args.training_size:
                                save_image(os.path.join(small_dir, folder_names + "face_" + file_name + "_" + str(person_id) + ext), face, args)
                            else:
                                save_image(os.path.join(args.out_dir, folder_names + "face_" + file_name + "_" + str(person_id) + ext), face, args)

                            faces_extracted = faces_extracted + 1

                files_processed = files_processed + 1

    exec_time = time.time() - start_time
    print(f"  Processed {files_processed} files in {exec_time} seconds for an average of {exec_time/files_processed} sec/file.")
    print(f"  Extracted {people_extracted} images of people and {faces_extracted} images of faces.")        


if __name__ == "__main__":
    main()
