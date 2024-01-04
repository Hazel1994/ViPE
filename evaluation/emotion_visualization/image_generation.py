import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from datasets import load_dataset
import json
from img_tools import generate_images
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generating images from ViPE and Chatgpt prompts")

    parser.add_argument(
        "--model_name", type=str, default='vipe', help="which model's prompts to use? [pass chatgpt or vipe']"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for image generation using Stable Diffusion"
    )
    parser.add_argument(
        "--saving_dir", type=str, default='/graphics/scratch2/staff/hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_True/gpt2-medium_context_ctx_7_lr_5e-05-v4/',
        help="a directory to save the generated images"
    )
    args = parser.parse_args()
    return args

def main():
    args=parse_args()

    # Load the dataset
    dataset = load_dataset('dair-ai/emotion')

    model_name=args.model_name
    prompts_path = './prompts/vipe/' if model_name == 'vipe' else './prompts/chatgpt/'
    saving_dir=args.saving_dir

    if model_name =='vipe':
        saving_dir =saving_dir + 'vipe/images/'
    else:
        saving_dir =saving_dir + 'chatgpt/images/'

    # Create the directory if it does not exist
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    with open(prompts_path + 'vis_emotion_train') as file:
        text_train = json.load(file)
    with open(prompts_path + 'vis_emotion_valid') as file:
        text_valid = json.load(file)

    if model_name =='chatgpt':
        text_train=[i for i in text_train['text'] ]
        text_valid = [i for i in text_valid['text']]

    text_train.extend(text_valid)

    prompt_dict={i:p for i,p in enumerate(text_train)}

    print('generating  images')
    generate_images(prompt_dict=prompt_dict, saving_path=saving_dir, batch_size=args.batch_size, gpu=0)


if __name__ == "__main__":
    main()
