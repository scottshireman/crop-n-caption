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
import cv2
from PIL import Image


import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
from annoy import AnnoyIndex
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from itertools import compress


SUPPORTED_EXT = [".webp", ".jpg", ".png", ".jpeg", ".bmp", ".jfif"]

#Basic functions related to opening and manipulating files and folders

def open_image(image_file):
    
    return Image.open(image_file)


def get_caption_path(image_path):

    yaml_path = os.path.splitext(image_path)[0] + '.yaml'
    txt_path = os.path.splitext(image_path)[0] + '.txt'

    if os.path.exists(yaml_path):
        caption_path = yaml_path
    elif os.path.exists(yaml_path):
        caption_path = txt_path
    else:
        caption_path = None

    return caption_path


def move_image_and_caption(image_path, target_path, copy=False):
    
    caption_path = get_caption_path(image_path)

    if copy:
        shutil.copy(caption_path, target_path)
        new_path = shutil.copy(image_path, target_path)
    else:
        shutil.move(caption_path, target_path)
        new_path = shutil.move(image_path, target_path)
    
    return new_path


def prepare_subfolder(folder_path, subfolder_name, reset_files=True):

    subfolder_path = os.path.join(folder_path, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
    else:
        if reset_files:
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    shutil.move(os.path.join(subfolder_path, file), folder_path)

    return subfolder_path  


# Functions to create image map and embeddings that will be used throughout

def generate_embeddings_and_map(folder_path, image_map_file, embeddings_file, model, processor, device, batch_size):
    
    check_path = prepare_subfolder(folder_path, 'check', reset_files=True)

    image_map = get_image_map(folder_path, check_path)
    np.save(image_map_file, image_map)

    all_embeddings = generate_embeddings(image_map['id'], image_map['path'], model, processor, device, batch_size)
    np.save(embeddings_file, all_embeddings)

    return image_map, all_embeddings


def get_image_map(image_directory, check_path):

    images_to_paths = []
    face_mask = []
    body_mask = []

    base_folder_name = os.path.basename(image_directory).strip()

    for root, dirs, files in os.walk(image_directory, topdown=True):
        [dirs.remove(d) for d in list(dirs) if d in ['check','delete','blurry']]

        for file in tqdm(files, position=1, desc=f'Finding and identifying all image files', leave=False):
            file_split = os.path.splitext(file)
            if file_split[1] in SUPPORTED_EXT:

                image_path = os.path.join(root,file)

                caption_path = get_caption_path(image_path)

                if not caption_path:
                    images_to_paths.append(None)
                    face_mask.append(False)
                    body_mask.append(False)
                    continue

                #Read the caption
                with open(caption_path, 'r') as f :
                    caption_text = f.read()
                f.close()

                # Look for captions that don't have the base folder name as they probably don't contain the subject
                # Move them and their caption files to the check folder
                if not args.no_check:
                    if base_folder_name not in caption_text:
                        images_to_paths.append(move_image_and_caption(image_path, check_path))
                        face_mask.append(False)
                        body_mask.append(False)
                        continue

                #Now look at tags and filename to see if its a face shot mask it as a face
                if "tag: closeup of face" in caption_text or "tag: a head shot" in caption_text or "_face_" in image_path:
                    images_to_paths.append(image_path)
                    face_mask.append(True)
                    body_mask.append(False)
                    continue

                #Whatever is left must be a body image
                images_to_paths.append(image_path) 
                face_mask.append(False)
                body_mask.append(True)
                
    training_mask = np.full(len(face_mask), False)
    validation_mask = training_mask

    indices = np.arange(len(images_to_paths))
    image_map = np.rec.fromarrays((indices, images_to_paths, face_mask, body_mask, training_mask, validation_mask), names=('id', 'path', 'face', 'body', 'training', 'validation'))
    
    return image_map
                

def generate_embeddings(all_image_ids, images_to_paths, model, processor, device, batch_size):

    all_embeddings = []
    progress_bar = tqdm(total=len(all_image_ids), desc="Generating new CLIP embeddings", position=1, leave=False)

    for i in range(0, len(all_image_ids), batch_size):
        batch_image_ids, batch_images = process_image_batch(all_image_ids, i, batch_size, images_to_paths)
                  
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        all_embeddings.extend(outputs.cpu().numpy())
        progress_bar.update(len(batch_image_ids))

    progress_bar.close()
    
    return all_embeddings


def process_image_batch(all_image_ids, start_idx, batch_size, images_to_paths):
    batch_image_ids = all_image_ids[start_idx: start_idx + batch_size]
    batch_images = []

    for image_id in batch_image_ids:
        try:
            image = open_image(images_to_paths[image_id])
            batch_images.append(image)
        except OSError:
            print(f"\nError processing image {images_to_paths[image_id]}, marking as corrupted.")

    return batch_image_ids, batch_images


# Functions to cluster images based on embeddings

def build_image_clusters(all_image_ids, labels):
    image_id_clusters = defaultdict(set)

    for image_id, cluster_label in zip(all_image_ids, labels):
        image_id_clusters[cluster_label].add(image_id)

    return image_id_clusters


def cluster_images_by_kmeans(all_image_ids, all_embeddings, n_clusters):

    min_indices = []
    n_clusters = round(n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(all_embeddings)
    image_id_clusters = build_image_clusters(all_image_ids, kmeans.labels_)

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, all_embeddings)

    center_images_ids = [all_image_ids[i] for i in closest]

    return image_id_clusters, center_images_ids


# Function that actually creates the training and validation sets based on the image map and embeddings

def create_training_set(all_embeddings, image_map, max_images, validation_percent):

    pbar = tqdm(position=1, leave=False, total=5)

    # Create face training set
    pbar.set_description('Getting face embeddings')
    face_embeddings = all_embeddings[image_map['face']]
    face_images = image_map['id'][image_map['face']]
    pbar.update(1)

    pbar.set_description('Creating face training set')
    num_images = len(face_images)
    number_images_to_keep = round(max_images / (1 - validation_percent))
    
    if number_images_to_keep > num_images:
        number_images_to_keep = num_images
        for face_image in face_images:
            image_map['training'][face_image] = True

    else:
        image_id_clusters, cluster_centers = cluster_images_by_kmeans(face_images, face_embeddings, number_images_to_keep)
        for cluster_center in cluster_centers:
            image_map['training'][cluster_center] = True
    pbar.update(2)

    # Create body training set limiting size to size of face set
    pbar.set_description('Getting body embeddings')
    body_embeddings = all_embeddings[image_map['body']]
    body_images = image_map['id'][image_map['body']]
    pbar.update(3)

    pbar.set_description('Creating body training set')
    num_images = len(body_images)
    
    if number_images_to_keep > num_images:
        number_images_to_keep = num_images
        for body_image in body_images:
            image_map['training'][body_image] = True

    else:
        image_id_clusters, cluster_centers = cluster_images_by_kmeans(body_images, body_embeddings, number_images_to_keep)
        for cluster_center in cluster_centers:
            image_map['training'][cluster_center] = True
    pbar.update(4)
            
    #Now find the most representative images from the training set for the validation set if validation is > 0
    pbar.set_description('Creating validation set')
    if validation_percent > 0:
        training_embeddings = all_embeddings[image_map['training']]
        training_images = image_map['id'][image_map['training']]
      
        training_count = len(training_images)
        validation_count = round(training_count * validation_percent)

        image_id_clusters, cluster_centers = cluster_images_by_kmeans(training_images, training_embeddings, validation_count)
        for cluster_center in cluster_centers:
            image_map['validation'][cluster_center] = True
            image_map['training'][cluster_center] = False

    pbar.update(5)
    pbar.close()

    return image_map
    

# Function to process images in a specific folder

def process_folder(folder_path, training_path, validation_path, model, processor, device, args):

    embeddings_file = os.path.join(folder_path,'embeddings.npy')
    image_map_file = os.path.join(folder_path,'image_map.npy')
    
    batch_size = args.batch_size

    # Generate new embeddingsand image map if forced or if they don't exist on disk. Otherwise just load the existing ones.
    if args.force_regen_embeddings or not os.path.exists(embeddings_file) or not os.path.exists(image_map_file):
        image_map, all_embeddings = generate_embeddings_and_map(folder_path, image_map_file, embeddings_file, model, processor, device, batch_size)

    all_embeddings = np.load(embeddings_file)
    image_map = np.load(image_map_file)

    # Check to see if they match length, if not regenerate
    if len(all_embeddings) != len(image_map['id']):
        image_map, all_embeddings = generate_embeddings_and_map(folder_path, image_map_file, embeddings_file, model, processor, device, batch_size)
        all_embeddings = np.load(embeddings_file)
        image_map = np.load(image_map_file)

    # Double check image_map to make sure all expected image files still exists    
    for idx, image_path in enumerate(tqdm(image_map['path'], desc='Updating image_map for missing images', position=1, leave=False)):
        if not os.path.exists(image_path):
            image_map['path'][idx] = None


    # Update image_map with training and validation sets
    image_map = create_training_set(all_embeddings, image_map, args.max_face_images, args.validation)

    # Copy training and validation images to the appropriate output folders
    training_image_paths = image_map['path'][image_map['training']]
    for training_image in training_image_paths:
        move_image_and_caption(training_image, training_path, copy=True)

    if args.validation > 0:
        validation_image_paths = image_map['path'][image_map['validation']]
        for validation_image in validation_image_paths:
            move_image_and_caption(validation_image, validation_path, copy=True)


# Determine if we're processing one folder or many and then call function to process folder(s) accordingily

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    if not os.path.exists(args.training_path):
        print("Training output path doesn't exist")
        quit()

    if args.validation > 0 and not os.path.exists(args.validation_path):
        print("Validation output path doesn't exist")
        quit()
    

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    if args.folders:
        folders = next(os.walk(args.img_dir))[1]
        pbar = tqdm(folders, position=0, leave=True)
        for folder in pbar:
            pbar.set_description(f"Processing folder: {folder}")
            folder_path = os.path.join(args.img_dir, folder)

            #Create subfolders in training and validation output folders
            training_path = os.path.join(args.training_path, folder)
            if not os.path.exists(training_path):
                os.mkdir(training_path)

            if args.validation > 0:
                validation_path = os.path.join(args.validation_path, folder)
                if not os.path.exists(validation_path):
                    os.mkdir(validation_path)
            else:
                validation_path = training_path
            
            process_folder(folder_path, training_path, validation_path, model, processor, device, args)
            
        pbar.close()

    else:
        process_folder(args.img_dir, args.training_path, args.validation_path, model, processor, device, args)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",
        type=str,
        default="input",
        help="Path to images"
        )
    parser.add_argument("--training_path",
        type=str,
        default="training",
        help="Path to copy training images. If folders also specificed, will create a subfolder for each folder."
        )
    parser.add_argument("--validation_path",
        type=str,
        default="validation",
        help="Path to copy validation images. If folders also specificed, will create a subfolder for each folder."
        )
    parser.add_argument(
        "--folders",
        action="store_true",
        default=False,
        help="img_dir is a folder of folders to be done seperately (as opposed to one folder with images)."
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=192,
        help="Batch size for generating CLIP embeddings. Higher values will require more VRAM. (default: 192)"
        )
    parser.add_argument(
        "--max_face_images",
        type=int,
        default=None,
        help="Max number of face images to keep. Will keep the same or fewer body images as well. (default: no limit)"
        )
    parser.add_argument(
        "--validation",
        type=float,
        default=0.15,
        help="Size of validation set to create. Set to 0 to disable validation. (default: 0.15)"
        )
    parser.add_argument(
        "--force_regen_embeddings",
        action="store_true",
        default=False,
        help="Force regenerating new embeddings even if they already exist (default: no)."
        )
    parser.add_argument(
        "--no_check",
        action="store_true",
        default=False,
        help="Don't look for captions not matching folder name before generating embeddings. Won't check anyway if using existing embeddings."
        )


    args = parser.parse_args()

    print(f"** Extracting from files in: {args.img_dir}")
    main(args)
