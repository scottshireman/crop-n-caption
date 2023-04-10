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
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel

import torch
from  pynvml import *

import time
from colorama import Fore, Style
from clip_interrogator import Config, Interrogator, LabelTable, load_list


SUPPORTED_EXT = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp"]

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
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip_model, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"BLIP2 Model loaded: {model_name}")
    return processor, model

def get_replace_list(opt):
    
    with open(opt.find, 'r') as f:
       lines = f.read().split('\n')

    return lines

def main(args):

    start_time = time.time()
    files_processed = 0

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    dtype = torch.float32 if args.force_cpu else torch.float16

    #Open CLIP model
    print(f"Loading CLIP model . . .")
    ci = Interrogator(Config(clip_model_name=args.clip_model, caption_model_name=None))
    mediums_table = LabelTable(load_list(args.mediums), 'terms', ci)
    emotions_table = LabelTable(load_list(args.emotions), 'terms', ci)
    print(f"GPU memory used: {get_gpu_memory_map()} MB")

    #Open BLIP2 model
    start_blip_load = time.time()
    print(f"Loading BLIP2 model . . .")
    print(f"Loading BLIP2 model {args.blip_model} . . .")
    blip_processor, blip_model = create_blip2_processor(args.blip_model, device, dtype)
    print(f"Loaded BLIP2 model in {time.time() - start_blip_load} seconds.")
    print(f"GPU memory used: {get_gpu_memory_map()} MB")



    if args.replace is not None:
        find_list = get_replace_list(args)

    # os.walk all files in args.img_dir recursively
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            #get file extension
            ext = os.path.splitext(file)[1]
            if ext.lower() in SUPPORTED_EXT:
                full_file_path = os.path.join(root, file)
                image = Image.open(full_file_path)

                #query BLIP
                inputs = blip_processor(images=image, return_tensors="pt", max_new_tokens=args.max_new_tokens).to(device, dtype)

                generated_ids = blip_model.generate(**inputs)
                blip_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                if blip_caption.isalnum() is False:
                        clean_string = "".join(ch for ch in blip_caption if (ch.isalnum() or ch == " "))
                        blip_caption = clean_string

                if args.replace is not None:
                    for s in find_list:
                        if s in blip_caption:
                             blip_caption = blip_caption.replace(s, args.replace)

                #query CLIP
                clip_medium = mediums_table.rank(ci.image_to_features(image), top_count=1)[0]
                clip_emotion = emotions_table.rank(ci.image_to_features(image), top_count=1)[0]     

                print(f"file: {file}, caption: {blip_caption}, {clip_medium}, {clip_emotion}")
                
                # get bare name
                name = os.path.splitext(full_file_path)[0]
                #name = os.path.join(root, name)
                if not os.path.exists(name):
                    if args.yaml:
                        final_caption = "main_prompt: " + blip_caption + "\ntags:"
                        if clip_medium != "":
                            final_caption = final_caption + "\n  - tag: "+ clip_medium
                        if clip_emotion != "":
                            final_caption = final_caption + "\n  - tag: " + clip_emotion
                            
                        with open(f"{name}.yaml", "w") as f:
                            f.write(final_caption)
                    else:
                        final_caption = blip_caption
                        if clip_medium != "":
                            final_caption = final_caption + ", " + clip_medium
                        if clip_emotion != "":
                            final_caption = final_caption + ", " + clip_emotion
                            
                        with open(f"{name}.txt", "w") as f:
                            f.write(generated_text)

                files_processed = files_processed + 1

    exec_time = time.time() - start_time
    # print(f"  Processed {files_processed} files in {exec_time} seconds for an average of {exec_time/files_processed} sec/file.")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="input", help="Path to images")
    parser.add_argument("--blip_model", type=str, default="salesforce/blip2-opt-6.7b", help="model from huggingface, ex. 'salesforce/blip2-opt-2.7b'")
    parser.add_argument("--clip_model", type=str, default="ViT-L-14/openai", help="CLIP model, ex. 'ViT-L-14/openai'")
    parser.add_argument("--force_cpu", action="store_true", default=False, help="force using CPU even if GPU is available, may be useful to run huge models if you have a lot of system memory")
    parser.add_argument("--max_new_tokens", type=int, default=24, help="max length for generated captions")
    parser.add_argument("--find", type=str, nargs="?", const=True, default="data/female.txt", help="a txt file containing one entry per line of strings to replace")
    parser.add_argument("--replace", type=str, nargs="?", required=False, const=True, default=None, help="string to replace with, ex. 'john doe'")
    parser.add_argument("--yaml", action="store_true", default=False, help="write yaml files instead of txt")
    parser.add_argument("--mediums", type=str, nargs="?", required=False, const=True, default='data/mediums.txt', help="text file with list of mediums for tags")
    parser.add_argument("--emotions", type=str, nargs="?", required=False, const=True, default='data/emotions.txt', help="text file with list of emotions for tags")

    args = parser.parse_args()

    print(f"** Using modela: {args.blip_model} and {args.clip_model}")
    print(f"** Captioning files in: {args.img_dir}")
    main(args)
