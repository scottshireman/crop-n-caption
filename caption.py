"""
Copyright [2022-2023] Scott Shireman

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

from PIL import Image
import argparse
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification

import torch
from  pynvml import *

import time
from colorama import Fore, Style
from unidecode import unidecode

# from clip_interrogator import Config, Interrogator, LabelTable, load_list



SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

def get_args(**parser_kwargs):
    """Get command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="output",
        help="Path to input images. (default: 'output')"
        )
    parser.add_argument(
        "--blip_model",
        type=str,
        default="salesforce/blip2-opt-6.7b",
        help="BLIP2 moodel from huggingface. You will need to use smaller model for <24GB VRAM (default: salesforce/blip2-opt-6.7b)"
        )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="Force using CPU for BLIP even if GPU is available. You need at least 24GB VRAM to use GPU and 64GB to use CPU but it will likely be very slow. I have not tested this."
        )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="In theory this should change length of generated captions. Don't lower it until you're seeing issues with too long captions. (default: 48)"
        )
    parser.add_argument(
        "--find",
        type=str,
        nargs="?",
        const=True,
        default="data/female.txt",
        help="A txt file containing one entry per line of strings to replace in the BLIP caption. Only works if --replace is also specific, ignored otherwise. (default: data/female.txt)"
        )
    parser.add_argument(
        "--replace",
        type=str,
        nargs="?",
        required=False,
        const=True,
        default=None,
        help="A string that will be used to replace the entries specific in the --find file, ex. 'john doe' (default: None)"
        )
    parser.add_argument(
        "--yaml",
        action="store_true",
        default=False,
        help="Write yaml files instead of txt. Recomended if your trainer supports it for flexibility later."
        )
    parser.add_argument(
        "--replace_from_folder",
        action="store_true",
        default=False,
        help="Use the folder name as the replace text and ignore the --replace flag."
        )
    parser.add_argument(
        "--tags_from_filename",
        action="store_true",
        default=False,
        help="Extract '_' delimitted tags from filename. Usual if subfolders were written to filenames with crop.py."
        )

    args = parser.parse_args()

    return args

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used/1024/1024

def create_blip2_processor(model_name, device, dtype=torch.float16):
    processor = Blip2Processor.from_pretrained( model_name )
    model = Blip2ForConditionalGeneration.from_pretrained( model_name, torch_dtype=dtype )
    model.to(device)
    model.eval()
    print(f"BLIP2 Model loaded: {model_name}")
    return processor, model

def create_clip_processor(model_name, device):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    
    model.to(device)
    model.eval()
    print(f"CLIP Model loaded: {model_name}")
    return processor, model

def get_lines_from_text(text_file_path):
    
    with open(text_file_path, 'r') as f:
       text = f.read().lower()
       lines = text.split('\n')

    return lines

def get_topk_clip_matches(image, options_list, model, processor, topk=1, device="cuda"):

    result = []
    inputs = processor(text=options_list, images=image, return_tensors="pt", padding=True)
    inputs.to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    for index in probs.topk(topk)[1][0].tolist():
        result.append(options_list[index])

    rounded_probs = [ round(elem,2) for elem in probs.topk(topk)[0][0].tolist() ]

    return result, rounded_probs #clothing_list[probs.argmax()].lower(), probs.max()

def check_color(image, base_clothing, base_prob, clothing_colors, fashion_model, fashion_processor, topk=1, device="cuda"):
    color, color_prob = get_topk_clip_matches(image, [s + " " + base_clothing for s in clothing_colors], fashion_model, fashion_processor, topk, device)
    if color_prob[0] > 0.65:
        return color[0], color_prob[0]
    else:
        return base_clothing, base_prob

def detect_emotion(image, model, extractor):
    
    inputs = extractor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = logits.softmax(dim=1)

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label], round(probs[0][logits.argmax(-1).item()].item(), 2)

def main():

    args = get_args()
    print(f"** Captioning files in: {args.img_dir}")
    
    start_time = time.time()
    files_processed = 0

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = torch.float32 if args.force_cpu else torch.float16

    #Open Generic CLIP model
    print(f"\nLoading generic CLIP model: laion/CLIP-ViT-B-32-laion2B-s34B-b79K. . .")
    clip_processor, clip_model = create_clip_processor("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", device)
    print(f"Loaded. GPU memory used: {get_gpu_memory_map()} MB")


    #Open Fashion CLIP model
    print(f"\nLoading Fashion CLIP model . . .")
    fashion_processor, fashion_model = create_clip_processor("patrickjohncyh/fashion-clip", device)
    print(f"Loaded. GPU memory used: {get_gpu_memory_map()} MB")

    #Open emotion classifier
    print(f"\nLoading emotion classifier model . . .")
    extractor = AutoFeatureExtractor.from_pretrained("kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105179")
    model = AutoModelForImageClassification.from_pretrained("kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105179")
    print(f"GPU memory used: {get_gpu_memory_map()} MB")

    #Get lists needed for models
    clothing = get_lines_from_text('data/clothing_simple.txt')
    clothing_whole = get_lines_from_text('data/clothing_whole.txt')
    clothing_tops = get_lines_from_text('data/clothing_tops.txt')
    clothing_bottoms = get_lines_from_text('data/clothing_bottoms.txt')
    clothing_colors = get_lines_from_text('data/clothing_colors.txt')
    photo_types = get_lines_from_text('data/mediums.txt')

    #Open BLIP2 model
    start_blip_load = time.time()
    print(f"\nLoading BLIP2 model {args.blip_model} . . .")
    blip_processor, blip_model = create_blip2_processor(args.blip_model, device, dtype)
    print(f"GPU memory used: {get_gpu_memory_map()} MB")

    if args.replace is not None or args.replace_from_folder:
        find_list = get_lines_from_text(args.find)

    replace_text = args.replace

    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        if args.replace_from_folder:
            replace_text = os.path.basename(root)
        
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)

                #query BLIP
                inputs = blip_processor(images=image, return_tensors="pt").to(device, dtype)


                generated_ids = blip_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                blip_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                if blip_caption.isalnum() is False:
                        clean_string = "".join(ch for ch in blip_caption if (ch.isalnum() or ch == " "))
                        blip_caption = clean_string

                if args.replace is not None or args.replace_from_folder:
                    for s in find_list:
                        if s in blip_caption:
                             blip_caption = blip_caption.replace(s, replace_text)

                #query CLIP - Using CLIP until more precise classification model is available
                photo_type, photo_type_prob = get_topk_clip_matches(image, photo_types, clip_model, clip_processor, topk=1, device="cuda")

                if photo_type[0] == "a beach photo":
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a polaroid photo":
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a professional photo":
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "an out of focus photo":
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "an outdoor photo":
                    photo_type_caption = photo_type[0]            

                elif photo_type[0] == "a black and white photo" and photo_type_prob[0] > 0.68:
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a blurry face photo" and photo_type_prob[0] > 0.76:
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a head shot" and photo_type_prob[0] > 0.56:
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a naked photo" and photo_type_prob[0] > 0.96:
                    photo_type_caption = photo_type[0]

                elif photo_type[0] == "a naked photo" and photo_type_prob[0] > 0.35:
                    photo_type_caption = "an erotic photo"

                else:
                    photo_type_caption = "a photo"

 
                #Query Fashion CLIP
                whole, whole_prob = get_topk_clip_matches(image, clothing, fashion_model, fashion_processor, 1, device)
                clothes_caption = ""

                if whole[0] in "closeup of face" and whole_prob[0] >= 0.5:
                    clothes_caption = whole[0]

                elif whole[0] == "naked body" and whole_prob[0] > 0.9:
                    clothes_caption = whole[0]

                elif whole[0] in "bikini kimono pajamas cap and gown" and whole_prob[0] >= 0.9:
                    color, color_prob = check_color(image, whole[0], whole_prob[0], clothing_colors, fashion_model, fashion_processor, 1, device)
                    clothes_caption = color
                    
                elif whole[0] in clothing_whole and whole_prob[0] >= 0.7:
                    color, color_prob = check_color(image, whole[0], whole_prob[0], clothing_colors, fashion_model, fashion_processor, 1, device)
                    clothes_caption = color
                    
                elif whole[0] in "mini dress" and whole_prob[0] >= 0.5:
                    color, color_prob = check_color(image, whole[0], whole_prob[0], clothing_colors, fashion_model, fashion_processor, 1, device)
                    clothes_caption = color
                    
                else:
                    top, top_prob = get_topk_clip_matches(image, clothing_tops, fashion_model, fashion_processor, 1, device=device)
                    if top[0] == "naked breasts" and top_prob[0] >= 0.5:
                        clothes_caption = "naked breasts"
                        
                    elif top_prob[0] >= 0.7:
                        color, color_prob = check_color(image, top[0], top_prob[0], clothing_colors, fashion_model, fashion_processor, 1, device)
                        clothes_caption = color

                    bottom, bottom_prob = get_topk_clip_matches(image, clothing_bottoms, fashion_model, fashion_processor, 1, device)
                    if bottom_prob[0] >= 0.7:
                        color, color_prob = check_color(image, bottom[0], bottom_prob[0], clothing_colors, fashion_model, fashion_processor, 1, device)
                        if clothes_caption == "":
                            clothes_caption = color
                        else:
                            clothes_caption = ", ".join([clothes_caption, color])

                # Classify emotion
                emotion, emotion_prob = detect_emotion(image, model, extractor)
                emotion_caption = ""
    
                if emotion == "happy" and emotion_prob >= 0.5:
                    emotion_caption = emotion

                elif emotion == "sad" and emotion_prob >= 0.7:
                    emotion_caption = emotion

                elif emotion == "suprise" and emotion_prob >= 0.4:
                    emotion_caption = emotion
                    
                elif emotion == "fear" and emotion_prob >= 0.68:
                    emotion_caption = emotion

                elif emotion == "disgust" and emotion_prob >= 0.63:
                    emotion_caption = emotion
  
                elif emotion == "angry" and emotion_prob > 0.96:
                    emotion_caption = emotion


                # Remove any non-ASCII charatcers as they cause problems with ED2
                blip_caption = unidecode(blip_caption).strip()

                # Remove any leading or trailing white space from CLIP captions
                photo_type_caption = photo_type_caption.strip()
                emotion_caption = emotion_caption.strip()
                clothes_caption = clothes_caption.strip()
                

                #Build full caption string
                text_caption = blip_caption

                if photo_type_caption != "":
                    text_caption = text_caption + ", " + photo_type_caption

                if clothes_caption != "":
                    text_caption = text_caption + ", " + clothes_caption

                if emotion_caption != "":
                    text_caption = text_caption + ", " + emotion_caption

                if args.tags_from_filename:
                    for filename_tag in filename_tags:
                        if filename_tag != "":
                            filename_tag = unidecode(filename_tag)
                            text_caption = text_caption + ", " + filename_tag.strip()

                file_name_for_display = file.ljust(40)[:39]
    
                print(f"{file_name_for_display} caption: {text_caption}")
                
                # get bare name
                name = os.path.splitext(full_file_path)[0]
                #name = os.path.join(root, name)
                if not os.path.exists(name):
                    if args.yaml:
                        yaml_caption = "main_prompt: " + blip_caption + "\ntags:"

                        if photo_type_caption != "":
                            yaml_caption = yaml_caption + "\n  - tag: " + photo_type_caption

                        if clothes_caption != "":
                            clothes_caption_list = clothes_caption.split(', ')
                            for item in clothes_caption_list:
                                yaml_caption = yaml_caption + "\n  - tag: " + item.strip()

                        if emotion_caption != "":
                            yaml_caption = yaml_caption + "\n  - tag: " + emotion_caption

                        if args.tags_from_filename:
                            for filename_tag in filename_tags:
                                if filename_tag != "":
                                    yaml_caption = yaml_caption + "\n  - tag: " + filename_tag.strip()
                            
                        with open(f"{name}.yaml", "w") as f:
                            f.write(yaml_caption)
                    else:
                        with open(f"{name}.txt", "w") as f:
                            f.write(text_caption)

                files_processed = files_processed + 1

    exec_time = time.time() - start_time
    print(f"  Processed {files_processed} files in {exec_time} seconds for an average of {exec_time/files_processed} sec/file.")
    

if __name__ == "__main__":

    main()
