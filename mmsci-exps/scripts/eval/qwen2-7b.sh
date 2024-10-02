devices=6

export CUDA_VISIBLE_DEVICES=$devices

model=qwen2-vl-7b

cd ../..

# ### Matching
python run_matching.py --model_name $model --k 1 --setting 1
python run_matching.py --model_name $model --k 1 --setting 2
python run_matching.py --model_name $model --k 1 --setting 3

# ### Captioning
python run_captioning.py --model_name $model --k 1 --with_abstract False --with_content False
python run_captioning.py --model_name $model --k 1 --with_abstract True --with_content False


