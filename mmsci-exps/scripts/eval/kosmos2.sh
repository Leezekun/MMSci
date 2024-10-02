devices=0
export CUDA_VISIBLE_DEVICES=$devices

model=kosmos2

cd ../..

### Captioning
python run_captioning.py --model_name $model --k 3 --with_abstract False --with_content False
python run_captioning.py --model_name $model --k 3 --with_abstract True --with_content False

# # ### Matching
python run_matching.py --model_name $model --k 5 --setting 1
python run_matching.py --model_name $model --k 1 --setting 2
python run_matching.py --model_name $model --k 5 --setting 3