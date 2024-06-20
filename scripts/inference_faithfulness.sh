CUDA_VISIBLE_DEVICES=0 python evalalign/eval/test_faithfulness.py \
    --model-path Fudan-FUXI/evalalign-v1.0-13b \
    --images-dir ./PixArt-XL-2-1024-MS \
    --output-dir ./results_faithfulness