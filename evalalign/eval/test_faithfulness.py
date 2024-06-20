import os
import json
import torch
import pandas as pd
import numpy as np
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

    disable_torch_init()

    model_name = args.model_path.split("/")[-1]

    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, None, model_name
        )
      
    sum_res = {}
    save_res = {}

    img_nums = len(os.listdir(args.images_dir))
    for p in os.listdir(args.images_dir):
        imgp = os.path.join(args.images_dir,p)
        img_score = []
        question_info = { "body": "<image>\nAre there any issues with the [human/animals] body structure in the image, such as multiple arms, missing limbs or legs when not obscured, multiple heads, limb amputations, etc? \nOptions: -1.There are no human or animal body in the picture, 0.The body structure of the people or animals in the picture has a very grievous problem that is unbearable, 1.The body structure of the people or animals in the picture has some serious problems and is not acceptable, 2.The body structure of the people or animals in the picture has a slight problem that does not affect the senses, 3.The body structure of the people or animals in the picture is basically fine, with only a few flaws, 4.The body structure of the people or animals in the picture is completely fine and close to reality, the answer is",
                            "Hand": "<image>\nAre there any issues with the [human/animals] hands in the image, such as having more or less than five fingers when not obscured, broken fingers, disproportionate finger sizes, abnormal nail size proportions, etc? \nOptions: -1.No human or animal hands are shown in the picture, 0.The hand in the picture has a very grievous problem that is unbearable, 1.The hand in the picture has some serious problems and is not acceptable, 2.The hand in the picture has a slight problem that does not affect the senses, 3.The hand in the picture is basically fine, with only a few flaws, 4.The hands in the picture are completely fine and close to reality, the answer is",
                            "face": "<image>\nAre there any issues with [human/animals] face in the image, such as facial distortion, asymmetrical faces, abnormal facial features, unusual expressions in the eyes, etc? \nOptions: -1.There is no face of any person or animal in the picture, 0.The face of the person or animal in the picture has a very grievous problem that is unbearable, 1.The face of the person or animal in the picture has some serious problems and is not acceptable, 2.The face of the person or animal in the picture has a slight problem that does not affect the senses, 3.The face of the person or animal in the picture is basically fine, with only a few flaws, 4.The face of the person or animal in the picture is completely fine and close to reality, the answer is",
                            "object": "<image>\nAre there any issues or tentative errors with objects in the image that do not correspond with the real world, such as distortion of items, etc? \nOptions: 0.There are objects in the image that completely do not match the real world, which is very serious and intolerable, 1.There are objects in the image that do not match the real world, which is quite serious and unacceptable, 2.There are slightly unrealistic objects in the image that do not affect the senses, 3.There are basically no objects in the image that do not match the real world, only some flaws, 4.All objects in the image match the real world, no problem, the answer is",
                            "common": "<image>\nDoes the generated image contain elements that violate common sense or logical rules? \nOptions:  0.The image contains elements that violate common sense or logical rules, which is very grievous and intolerable, 1.The presence of elements in the image that seriously violate common sense or logical rules is unacceptable, 2.The image contains elements that violate common sense or logical rules, which is slightly problematic and does not affect the senses, 3.There are basically no elements in the image that violate common sense or logical rules, only some flaws, 4.There are no elements in the image that violate common sense or logical rules, and they are close to reality, the answer is"}
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

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images_tensor],
                    #image_sizes=image_sizes,
                    do_sample=False,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            output = tokenizer.decode(output_ids[0][1:]).strip()
            print("output",output)
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            scores = re.findall(r"-?\d+", output)
            if len(scores) > 0:
                score = int(scores[0])
            else:
                score = 0

            ans_typed[typed] = score
            if typed not in sum_res:
                sum_res[typed] = [score]
            else:
                sum_res[typed].append(score)
            img_score.append(score)
        print(imgp, img_score)

    grained_dict = {}
    sums = []
    sum_1 = []

    print(sum_res)
    nums = []
    for td, tdres in sum_res.items():
        num_1 = int(img_nums - sum(np.array(tdres)==-1))
        nums.append(num_1)
        #sums.append(sum(tdres))
        value_no = [v if v!=-1 else 0 for v in tdres]
        sum_1.append(sum(value_no))
        grained_dict[f"{td}_score"] = sum(value_no)
        grained_dict[f"{td}_num"] = num_1
        grained_dict[f"{td}_average"] = sum(value_no)/num_1 if num_1>0 else 0
        #print(td, sum(tdres))
    #print(sum(sums))
    #grained_dict["res"] = sum(sums)
    grained_dict["total_score"] = sum(sum_1)
    grained_dict["num"] = sum(nums)
    grained_dict["avg_score"] = sum(sum_1)/sum(nums) if sum(nums)>0 else 0
    print(grained_dict)
    #save_res[classd] = grained_dict
    #save_dir = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/yangxiaomeng/Code/AIGCbenchmark/benchmarkutils/inference_fidelity_model/prompt_pro/our_fidelity_test_result_fidelity_t2i"
    os.makedirs(args.output_dir,exist_ok=True)
    df = pd.DataFrame.from_dict(grained_dict,orient='index')
    df.to_excel(f"{args.output_dir}/result_test_faithfulness.xlsx")
    with open(f'{args.output_dir}/result_test_faithfulness.json', 'w', encoding='utf-8') as f:
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


