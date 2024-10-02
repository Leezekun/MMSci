for model in internvl2-1b internvl2-2b  # qwen2-vl-2b-mmsci-mixed  idefics3-8b llama3.2-11b  #   #  internvl2-8b minicpm #  claude-3-5-sonnet-20240620 qwen2-vl-2b qwen2-vl-7b   # gpt-4-turbo blip2 llava qwen llava-next llava-next-mistral
do
    CUDA_VISIBLE_DEVICES=1 python textgen_scores.py --model $model --k 3
done