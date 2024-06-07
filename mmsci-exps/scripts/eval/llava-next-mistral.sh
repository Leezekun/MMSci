devices=1
export CUDA_VISIBLE_DEVICES=$devices

model=llava-next-mistral

cd ../..

### Captioning
python run_captioning.py --model_name $model --k 3 --with_abstract False --with_content False
python run_captioning.py --model_name $model --k 3 --with_abstract True --with_content False

# ### Matching
python run_vqa.py --model_name $model --k 5 --setting 1
python run_vqa.py --model_name $model --k 5 --setting 2
python run_vqa.py --model_name $model --k 5 --setting 3