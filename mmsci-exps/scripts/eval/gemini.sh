cd ../..

for model in gemini-1.5-flash-001 gemini-1.5-pro-001 # 
do
    # Matching
    for setting in 1 2 3
    do
        python run_matching_api.py --model_name=$model \
                            --setting=$setting \
                            --k=1
   # Captioning
    python run_captioning_api.py --model_name $model --k 1 --with_abstract False --with_content False
    python run_captioning_api.py --model_name $model --k 1 --with_abstract True --with_content False
    
    done
done
