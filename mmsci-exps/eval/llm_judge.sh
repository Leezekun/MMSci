for model in kosmos2 # qwen2-vl-2b-mmsci-mixed # idefics3-8b llama3.2-11b # gemini-1.5-pro-002 claude-3-5-sonnet-20240620 qwen2-vl-2b qwen2-vl-7b  kosmos2 idefics2-8b  internvl2-1b internvl2-2b internvl2-8b minicpm  # gpt-4-turbo # blip2 llava qwen llava-next llava-next-mistral
do
    python llm_judge_scores.py --model $model --evaluator openai
done