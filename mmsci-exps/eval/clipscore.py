'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import pprint
import warnings
from packaging import version
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument(
        '--image_dir',
        default='/home/ubuntu/MMSci/mmsci-data/benchmark/dev/images',
        type=str,
        help='Directory of images, with the filenames as image ids.')
    
    parser.add_argument(
        '--base_generation_output_dir',
        default='output/image_caption_generation/',
        type=str,
    )

    parser.add_argument(
        '--k',
        default=3, 
        type=int
    )

    parser.add_argument(
        '--base_score_dir', 
        default='eval_scores/image_caption_generation/clipscore',
        type=str,
    )

    parser.add_argument(
        '--overwrite',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--references_json',
        default=None,
        help='Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():

    print(f'Evaluating {args.input_json}')

    score_dir = os.path.join(args.base_score_dir, args.tag)
    os.makedirs(score_dir, exist_ok=True)
    score_file = os.path.join(score_dir, args.input_json.split('/')[-1])
    print(score_file)
    if os.path.exists(score_file) and not args.overwrite:
        print(f'Already evaluated. Will skip...')
        return

    with open(args.input_json) as f:
        data = json.load(f)

    # reformat data
    old_candidates = {}
    old_references = {}
    for item in data:
        if 'prediction' not in item:
            continue
        image_id = item['image'].split('.')[0]
        old_candidates[image_id] = item['prediction']
        old_references[image_id] = [item['caption']]

    candidates = []
    references = []
    image_ids = old_references.keys()
    for cid in image_ids:
        if cid in old_candidates:
            candidates.append(old_candidates[cid][args.eval_pred_idx])
            references.append(old_references[cid])
    image_paths = [os.path.join(args.image_dir, f'{cid}.png') for cid in image_ids]

    if isinstance(references[0], str):
        references = [[r] for r in references]
            

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    # get text-text clipscore
    _, per_instance_text_text = get_refonlyclipscore(
        model, references, candidate_feats, device)
    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
    scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                for image_id, clipscore, refclipscore in
                zip(image_ids, per_instance_image_text, refclipscores)}

    avg_scores = {}
    clipscore = np.mean([s['CLIPScore'] for s in scores.values()])
    refclipscore = np.mean([s['RefCLIPScore'] for s in scores.values()])
    print('CLIPScore: {:.4f}'.format(clipscore))
    print('RefCLIPScore: {:.4f}'.format(refclipscore))
    avg_scores['CLIPScore'] = clipscore
    avg_scores['RefCLIPScore'] = refclipscore

    if args.save_per_instance:
        with open(args.save_per_instance, 'w') as f:
            f.write(json.dumps(scores))

    # scores = json.load(open(score_file, 'r'))
    # scores['CLIPScore'] = clipscore
    # scores['RefCLIPScore'] = refclipscore
    # with open(score_file, 'w') as fout:
    #     json.dump(avg_scores, fout, indent=4)

    return score_file, avg_scores

if __name__ == '__main__':

    args = parse_args()
    args.compute_other_ref_metrics = 1
    args.overwrite = 1

    base_output_dir = '/home/ubuntu/MMSci/mmsci-exps/eval/output/image_caption_generation'

    for w_abs in [False, True]:
        for w_ctx in [False, True]:
            if w_abs and w_ctx:
                continue
            tag = f'abs{w_abs}_ctx{w_ctx}'
            args.tag = tag
            k = 3 # inference times
            file = f"{args.model}.json"
            args.input_json = os.path.join(base_output_dir, tag, f"k_{k}", file)

            try:
                all_scores = defaultdict(list)
                for i in range(k):
                    args.eval_pred_idx = i
                    score_file, scores = main()
                    for metric, score in scores.items():
                        all_scores[metric].append(score)
                with open(score_file, 'w') as fout:
                    json.dump(all_scores, fout, indent=4)
            except Exception as e:
                print(e)
