import os
import json
import torch
import pandas as pd
from evalalign.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import argparse

from evalalign.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from evalalign.conversation import conv_templates, SeparatorStyle
from evalalign.model.builder import load_pretrained_model
from evalalign.utils import disable_torch_init
from evalalign.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    prompt_json = "./configs/test_alignment_prompt.json"
    id2prompt = {}
    with open(prompt_json,"r") as f:
        datasets_prompt = json.load(f)
        for data in datasets_prompt:
            prompts_id = data["prompt_id"]
            id2prompt[prompts_id] = data["question_info"]
    
    model_name = args.model_path.split("/")[-1]
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, None, model_name
        )
    
    img_nums = len(os.listdir(args.images_dir))
            #i = 0
    sum_res = {}
    for p in os.listdir(args.images_dir):

        imgp = os.path.join(args.images_dir,p)
        prompt_id = p.split(".")[0]

        if prompt_id not in id2prompt:
            raise ValueError("prompt_id must be used to name the generated image! The name of the image must match the promt_id!")
        question_info = id2prompt[prompt_id]
        ans_typed = {}

        for typed, question in question_info.items():

            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in question:
                if model.config.mm_use_im_start_end:
                    question = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
                else:
                    question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
            else:
                if model.config.mm_use_im_start_end:
                    question = image_token_se + "\n" + question
                else:
                    question = question
            
            conv_mode = "v1" 

            if "evalalign-v1.0-34b" in model_name:
                conv_mode = "mpt"
            elif "evalalign-v1.0-13b" in model_name:
                conv_mode = "v1" 

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images = [load_image(imgp)]

            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            #print(input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    #image_sizes=image_sizes,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            output = tokenizer.decode(output_ids[0][1:]).strip()
            print("output",output)
            scores = re.findall(r'\d+', output)
            if len(scores) > 0:
                score = int(scores[0])
            else:
                score = 0
            ans_typed[typed] = score
            if typed not in sum_res:
                sum_res[typed] = [score]
            else:
                sum_res[typed].append(score)

    grained_dict = {}
    sums = []
    for td, tdres in sum_res.items():
        grained_dict[f"{td}_score"] = sum(tdres)
        grained_dict[f"{td}_num"] = len(tdres)
        grained_dict[f"{td}_avgerage"] = sum(tdres)/len(tdres) if len(tdres) >0 else 0

        sums.append(sum(tdres))
        print(td, sum(tdres))
    grained_dict[f"total_score"] = sum(sums)
    grained_dict[f"total_avgerage"] = sum(sums)/img_nums 

    os.makedirs(args.output_dir,exist_ok=True)
    df = pd.DataFrame.from_dict(grained_dict,orient='index')
    df.to_excel(f"{args.output_dir}/result_test_alignment.xlsx")
    with open(f'{args.output_dir}/result_test_alignment.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(grained_dict,indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="sais/evalalign-v1.0-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--images-dir", type=str,default=None)
    parser.add_argument("--output-dir", type=str,default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

