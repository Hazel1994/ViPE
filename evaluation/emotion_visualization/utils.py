import json
import os

import evaluate
import torch
import torch.nn.functional as F
from bert_score import score
from datasets import DatasetDict, load_dataset
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from torchvision.datasets.folder import default_loader

def update_dataset_chatgpt(dataset, prompt_list):
    new_dataset = {k: [] for k in dataset['train'].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset['train']):
        example_copy = example.copy()
        example_copy['hypothesis'] = prompt_list[idx]

        for k, v in example_copy.items():
            new_dataset[k].append(v)
    new_dataset = Dataset.from_dict(new_dataset)
    new_dataset = DatasetDict({'train': new_dataset})
    return new_dataset


def get_batch(data, batch_size):
    for num, index in enumerate(range(0, len(data), batch_size)):
        if (num + 1) * batch_size < len(data):
            samples = data[index:(num + 1) * batch_size]
        else:
            samples = data[index:]

        yield samples


def save_s_json(path, name, data):
    with open(path + name, 'w') as file:
        json.dump(data, file, indent=4)


def visualizer(text, model, tokenizer, device, do_sample, epsilon_cutoff=.0001, temperature=1):
    text = [tokenizer.eos_token + i + tokenizer.eos_token for i in text]
    batch = tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    max_prompt_length = 50
    # max_length=input_ids.shape[1] + max_prompt_length
    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_prompt_length,
                                   do_sample=do_sample, epsilon_cutoff=epsilon_cutoff, temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:, -(generated_ids.shape[1] - input_ids.shape[1]):],
                                       skip_special_tokens=True)

    return pred_caps


def get_chatgpt_text(path_to_jsons):
    hypothesis = []

    for fine_index, filename in enumerate(os.listdir(path_to_jsons)):
        if filename.endswith('.json'):
            file_path = os.path.join(path_to_jsons, filename)
            with open(file_path) as file:
                json_data = json.load(file)

            prompts = json_data['choices'][0]['message']['content'].split('\n')

            number_of_prompts = 50

            for line_number in range(number_of_prompts):
                if len(prompts) > line_number and len(prompts[line_number].split('.')) > 1:
                    hypothesis.append(prompts[line_number].split('.')[1])

                elif len(prompts) > line_number:
                    hypothesis.append(prompts[line_number])

                elif len(
                        prompts) <= line_number:  # in case if chatgpt's output does not match the number of given prompts
                    hypothesis.append('noise')

    return hypothesis


def update_dataset_chatgpt(dataset, prompt_list, portion):
    new_dataset = {k: [] for k in dataset[portion].features.keys()}
    # Iterate over the dataset examples
    for idx, example in enumerate(dataset[portion]):
        example_copy = example.copy()
        example_copy['text'] = prompt_list[idx]

        for k, v in example_copy.items():
            new_dataset[k].append(v)
    return new_dataset


class SingleCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collator(self, batch):
        text_b = []
        labels_b = []
        for sample in batch:
            text = sample['text']
            label = sample['label']
            labels_b.append(label)
            text_b.append(text)

        tokens = self.tokenizer(text_b, padding=True, return_token_type_ids=False, return_tensors="pt")
        tokens['labels'] = torch.Tensor(labels_b)
        return tokens

    def __call__(self, batch):
        return self.collator(batch)


# Create a custom dataset for loading images and captions
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, captions_dict):
        self.image_folder = image_folder
        self.captions_dict = captions_dict
        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = f"{self.image_folder}/{image_id}.png"
        image = default_loader(image_path)
        caption = self.captions_dict[image_id]

        return image, caption


class DataCollator:

    def __init__(self, processor):
        self.processor = processor

    def collator(self, batch):
        image_b = []
        caption_b = []
        for image, caption in batch:
            image_b.append(image)
            caption_b.append(caption)

        inputs = self.processor(text=caption_b, images=image_b, return_tensors="pt", padding=True, max_length=77,
                           truncation=True)

        return inputs

    def __call__(self, batch):
        return self.collator(batch)
