from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import json

def write_list_to_json(data_list, filename):
    with open(filename, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

def prepare_ViPE(args,model_type):
    if model_type=='vipe':
        model = GPT2LMHeadModel.from_pretrained(args.vipe_path)
    else:
        model = GPT2LMHeadModel.from_pretrained('fittar/ViPE-M-CTX7')

    model.to(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def generate_from_tokens(batch, model, tokenizer,device,do_sample,top_k=100, epsilon_cutoff=.00005, temperature=1):
    # text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    # batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    max_new_tokens=30

    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample,top_k=top_k, epsilon_cutoff=epsilon_cutoff, temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:, -(generated_ids.shape[1] - input_ids.shape[1]):], skip_special_tokens=True)


    return pred_caps


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

class collator_cl:
    def __init__(self,tokenizer, vipe=True):
        self.tokenizer=tokenizer
        self.vipe=vipe

    def __call__(self, batch):
        # Tokenize sentences
        batch=[str(i) for i in batch]
        if self.vipe:
            batch = [self.tokenizer.eos_token + i + self.tokenizer.eos_token for i in batch]

        encoded_inputs = self.tokenizer(batch, padding=True, return_tensors="pt")
        return encoded_inputs
