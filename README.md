
# EvalAlign

# EVALALIGN: Supervised Fine-Tuning Multimodal LLMs with Human-Aligned Data for Evaluating Text-to-Image Models

## Contents
- [Install](#install)
- [EvalAlign Weights](#evalalign-weights)
- [Evaluation](#evaluation)

## Install
1. Clone this repository and navigate to EvalAlign folder
```bash
git clone https://github.com/SAIS-FUXI/EvalAlign.git
cd EvalAlign
```
2. Install Package
```Shell
conda create -n evalalign python=3.10 -y
conda activate evalalign
pip install --upgrade pip 
pip install -e .
```

## EvalAlign Weights
Please download the [Model weight](https://huggingface.co/Fudan-FUXI/evalalign-v1.0-13b) from huggingface.

## Evaluation
### Image Faithfulness evaluation
You must use the [prompt](https://github.com/SAIS-FUXI/EvalAlign/configs/test_faithfulness_prompt.json) provide about faithfulness to generate some images on your own model or open source model.The file name of the image needs to be consistent with prompt_id.
```shell
  {
    "prompt_id": "259_2_2",
    "prompt": "A young man was painting a beautiful landscape with a yellow brush and a black canvas."
  }
```
For example, in this data, you generated an image using prompt and named it "259_2_2. jpg".
```shell
#Run script
./scripts/inference_faithfulness.sh
```
You need to modify the path in the script
```shell
CUDA_VISIBLE_DEVICES=0 python evalalign/eval/test_faithfulness.py \
    --model-path Fudan-FUXI/evalalign-v1.0-13b \ # Downloaded model weights
    --images-dir ./PixArt-XL-2-1024-MS \  # The folder for generating images
    --output-dir ./results_faithfulness 
```
- result faithfulness

You will get a body, hand，face，object, common, The scores of the five dimensions and the average score of the overall model
```shell
{
  "body_score": 217,
  "body_num": 100,
  "body_average": 2.17,
  "Hand_score": 60,
  "Hand_num": 89,
  "Hand_average": 0.6741573033707865,
  "face_score": 137,
  "face_num": 81,
  "face_average": 1.691358024691358,
  "object_score": 250,
  "object_num": 100,
  "object_average": 2.5,
  "common_score": 105,
  "common_num": 100,
  "common_average": 1.05,
  "total_score": 769,
  "num": 470,
  "avg_score": 1.6361702127659574
}
```
### Text-to-Image Alignment evaluation
Same as Faithfulness.You must use the [prompt](https://github.com/SAIS-FUXI/EvalAlign/configs/test_alignment_prompt.json) provide about faithfulness to generate some images on your own model or open source model.The file name of the image needs to be consistent with prompt_id.
```shell
  {
    "prompt_id": "99",
    "prompt": "two refrigerators stand side-by-side in a kitchen, with two potted plants on either side of them."
  }
```
For example, in this data, you generated an image using prompt and named it "99. jpg".
```shell
#Run script
./scripts/inference_alignment.sh
```
You need to modify the path in the script
```shell
CUDA_VISIBLE_DEVICES=0 python evalalign/eval/test_faithfulness.py \
    --model-path Fudan-FUXI/evalalign-v1.0-13b \ # Downloaded model weights
    --images-dir ./IF-I-XL-v1.0 \  # The folder for generating images
    --output-dir ./results_alignment 
```
- result faithfulness
You will get a Object, Count，Spatial，Action, Color, Style.The scores of the six dimensions and the average score of the overall model.
```shell
{
  "Object_score": 209,
  "Object_num": 118,
  "Object_avgerage": 1.771186440677966,
  "Count_score": 160,
  "Count_num": 109,
  "Count_avgerage": 1.4678899082568808,
  "Spatial_score": 155,
  "Spatial_num": 85,
  "Spatial_avgerage": 1.8235294117647058,
  "Action_score": 102,
  "Action_num": 54,
  "Action_avgerage": 1.8888888888888888,
  "Color_score": 51,
  "Color_num": 26,
  "Color_avgerage": 1.9615384615384615,
  "Style_score": 50,
  "Style_num": 25,
  "Style_avgerage": 2.0,
  "total_score": 727,
  "total_avgerage": 6.058333333333334
}
```



## Citation
```bibtex
@article{tan2024evalalign,
  title={EVALALIGN: Supervised Fine-Tuning Multimodal LLMs with Human-Aligned Data for Evaluating Text-to-Image Models},
  author={Tan, Zhiyu and Yang, Xiaomeng and Qin, Luozheng and Yang, Mengping and Zhang, Cheng and Li, Hao},
  journal={arXiv preprint arXiv:??},
  year={2024},
  institution={Shanghai Academy of AI for Science and Carnegie Mellon University and Fudan University},
}
```
## Acknowledgement
- [Llava](https://github.com/haotian-liu/LLaVA): Our model is trained on llava and has excellent multimodal reasoning ability！

