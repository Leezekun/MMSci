import os
import json
import pickle
import argparse
from tqdm import tqdm
from subjects import subjects


def split_data_to_save(args, category, data):

    if args.text_only:
        dir_name = f"{category}-text-only"
    else:
        dir_name = f"{category}"
    save_path = os.path.join(out_dir, dir_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    shard_idx = 0
    shard_cnt = 0
    shard_data = []

    for dp in tqdm(data):

        # check if valid
        if not args.text_only:
            
            valid = True

            # check data
            image_info = dp["image_info"]
            similarity_matrix = dp["similarity_matrix"]
            text_list = dp["text_list"]

            # check text list
            if len(text_list) == 0:
                valid = False
            for sent in text_list:
                if not isinstance(sent, str):
                    print(sent)
                    valid = False
                    break

            # check image list
            # if len(image_info) == 0:
            #     valid = False
            for image in image_info:
                """
                [
                    "face_detections", "index", "image_name", "local_path", "raw_url", 
                    "caption", "image_base64", "matched_text_index", "mathced_sim"
                ]
                """
                try:
                    if not image["image_base64"]:
                        valid = False
                        break
                    if not image["matched_text_index"]:
                        valid = False
                        break
                except Exception as e:
                    print(e)
                    valid = False
                    break
            
            if valid:
                # split into shared
                if shard_cnt < args.shard_size:
                    shard_data.append(dp)
                    shard_cnt += 1
                else:
                    processed_data_path = os.path.join(save_path, f"shard_{shard_idx}.pkl")
                    processed_data_count_path = os.path.join(save_path, f"shard_{shard_idx}.count")
                    with open(processed_data_path, "wb") as file:
                        pickle.dump(shard_data, file) 
                    with open(processed_data_count_path, "w") as file:
                        file.write(str(shard_cnt))
                    shard_idx += 1
                    shard_cnt = 1
                    shard_data = [dp]

        # remove the added image captions in the sentences
        else: 
            # check data
            image_info = dp["image_info"]
            similarity_matrix = dp["similarity_matrix"]
            text_list = dp["text_list"]

            # check text list
            if len(image_info) > 0:
                matched_text_indices = []
                for image in image_info:
                    """
                    [
                        "face_detections", "index", "image_name", "local_path", "raw_url", 
                        "caption", "image_base64", "matched_text_index", "mathced_sim"
                    ]
                    """
                    matched_text_index = image["matched_text_index"]
                    if matched_text_index < len(text_list):
                        matched_text_indices.append(matched_text_index)
                matched_text_indices.sort(reverse=True)
                for pos in matched_text_indices:
                    text_list.pop(pos)

                # delete the images
                dp["image_info"] = []
                dp["similarity_matrix"] = []
                dp["text_list"] = text_list

            # split into shared
            if shard_cnt < args.shard_size:
                shard_data.append(dp)
                shard_cnt += 1
            else:
                processed_data_path = os.path.join(save_path, f"shard_{shard_idx}.pkl")
                processed_data_count_path = os.path.join(save_path, f"shard_{shard_idx}.count")
                with open(processed_data_path, "wb") as file:
                    pickle.dump(shard_data, file) 
                with open(processed_data_count_path, "w") as file:
                    file.write(str(shard_cnt))
                shard_idx += 1
                shard_cnt = 1
                shard_data = [dp]


if __name__ == '__main__':
    
    sample_per_shard = 200

    base_path = "../rawdata"
    out_dir = f"../../VILA/playground/data/mmsci"

    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--category', type=str, default="all") #
    parser.add_argument('--text_only', action="store_true", default=False) #
    parser.add_argument('--shard_size', type=int, default=200) #

    args, unknown = parser.parse_known_args()
    print(args)

    mmsci_data = []

    all_categories = list(subjects.keys())
    if args.category == "all":
        scraped_categories = all_categories
    else:
        assert args.category in all_categories
        scraped_categories = [args.category]

    for category in scraped_categories:
        category_name = category.split()[0].lower()

        # load training data
        processed_data_path = os.path.join(base_path, category, f"mmsci_{category_name}_train_data.pkl")
        if os.path.exists(processed_data_path):
            with open(processed_data_path, "rb") as file:
                category_data = pickle.load(file)
            mmsci_data.extend(category_data)
        else:
            raise NotImplementedError
    
        # split the data into different shards
        if args.text_only:
            dir_name = f"{category_name}-text-only"
        else:
            dir_name = f"{category_name}"
        save_path = os.path.join(out_dir, dir_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # split and save
        split_data_to_save(args, category_name, category_data)
    
    if args.category == "all":
        # split and save
        split_data_to_save(args, "all", mmsci_data)

    
    


        

