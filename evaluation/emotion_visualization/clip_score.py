import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
from utils import save_s_json
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPModel
import torch.nn.functional as F
from utils import ImageCaptionDataset, DataCollator
import argparse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Generating images from ViPE and Chatgpt prompts")

    parser.add_argument(
        "--model_name", type=str, default='vipe', help="which model's prompts to use? [pass chatgpt or vipe']"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for computing the clip scores"
    )
    parser.add_argument(
        "--saving_dir", type=str, default='/graphics/scratch2/staff/hassan/checkpoints/lyrics_to_prompts/vis_emotion/',
        help="the saving_dir used for image_generation.py"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model_name = args.model_name
    prompts_path = './prompts/vipe/' if model_name == 'vipe' else './prompts/chatgpt/'
    saving_dir = args.saving_dir

    if model_name == 'vipe':
        saving_dir=saving_dir + 'vipe/'
        clip_dir = saving_dir + 'clip_score/'
    else:
        saving_dir = saving_dir + 'chatgpt/'
        clip_dir = saving_dir + 'clip_score/'

    os.makedirs(clip_dir, exist_ok=True)

    with open(prompts_path + 'vis_emotion_train') as file:
        text_train = json.load(file)
    with open(prompts_path + 'vis_emotion_valid') as file:
        text_valid = json.load(file)

    text_train.extend(text_valid)

    captions_dict = {i: p for i, p in enumerate(text_train)}

    clip_model = 'openai/clip-vit-large-patch14'
    processor = AutoProcessor.from_pretrained(clip_model)

    # Create the dataset and data loader
    dataset = ImageCaptionDataset(saving_dir + 'images', captions_dict)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DataCollator(processor))

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained(clip_model)
    model.to(device)
    all_similarities = []

    # Compute CLIP score for each image-caption pair
    for num, inputs in enumerate(tqdm(dataloader)):

        # Move inputs to the device (GPU if available)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Compute CLIP scores
        outputs = model(**inputs)
        # Compute the cosine similarity between each image and text embedding
        cos_similarities = F.cosine_similarity(outputs.text_embeds, outputs.image_embeds, dim=1)
        logits_per_image = outputs.logits_per_image

        all_similarities.extend(cos_similarities.tolist())

    save_s_json(clip_dir, 'clip_scores', all_similarities)
    save_s_json(clip_dir, 'clip_scores_results', {'mean': np.mean(all_similarities), 'std': np.mean(all_similarities)})
    print("mean:", np.mean(all_similarities))
    print("std:", np.std(all_similarities))

if __name__ == "__main__":
    main()
