import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import Dataset
from utils import visualizer, get_batch, save_s_json, get_chatgpt_text, update_dataset_chatgpt
import json
from utils import SingleCollator
import torch.nn.functional as F
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default='vipe', help="which model to use? [pass chatgpt or vipe']"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for prompt generation using ViPE"
    )
    args = parser.parse_args()
    return args


def main():
    # Load the dataset
    dataset = load_dataset('dair-ai/emotion')
    # Split the dataset into train and validation sets
    train_dataset = dataset['train']
    valid_dataset = dataset['test']

    device = 'cuda'
    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size

    saving_dir = './prompts/vipe/' if model_name == 'vipe' else './prompts/chatgpt/'

    if model_name == 'chatgpt':
        # get chat gpt prompts
        prompt_list_valid = get_chatgpt_text(saving_dir + 'valid_jsons/')
        # update the dataset
        dataset_valid = update_dataset_chatgpt(dataset, prompt_list_valid, 'test')

        save_s_json(saving_dir, 'vis_emotion_valid', dataset_valid)
        print('saved valid data')

        # get chat gpt prompts from the jsons output
        prompt_list_train = get_chatgpt_text(saving_dir + 'train_jsons/')
        # update the dataset, whatever that means
        dataset_train = update_dataset_chatgpt(dataset, prompt_list_train, 'train')

        save_s_json(saving_dir, 'vis_emotion_train', dataset_train)
        print('saved train data')


    elif not os.path.exists(saving_dir + 'vis_emotion_train'):

        model = GPT2LMHeadModel.from_pretrained('fittar/ViPE-M-CTX7')
        model.to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        tokenizer.pad_token = tokenizer.eos_token

        text_train = []
        for num, batch in enumerate(get_batch(train_dataset['text'], batch_size)):
            if num % 50:
                print(num, 'out of ', len(train_dataset['text']) / batch_size)
            text_train.extend(visualizer(batch, model, tokenizer, device, False, epsilon_cutoff=.0005, temperature=1.1))

        save_s_json(saving_dir, 'vis_emotion_train', text_train)
        print('saved training data')

        text_valid = []
        for batch in get_batch(valid_dataset['text'], batch_size):
            text_valid.extend(visualizer(batch, model, tokenizer, device, False, epsilon_cutoff=.0005, temperature=1.1))
        save_s_json(saving_dir, 'vis_emotion_valid', text_valid)
        print('saved valid data')

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(saving_dir + 'vis_emotion_train') as file:
        text_train = json.load(file)
    with open(saving_dir + 'vis_emotion_valid') as file:
        text_valid = json.load(file)

    if model_name == 'vipe':
        # Create a new dataset with new data
        vis_valid_dataset = Dataset.from_dict({'text': text_valid, 'label': valid_dataset['label']})
        vis_train_dataset = Dataset.from_dict({'text': text_train, 'label': train_dataset['label']})
    else:
        # Create a new dataset with new data
        vis_valid_dataset = Dataset.from_dict(text_valid)
        vis_train_dataset = Dataset.from_dict(text_train)

    train_dataloader = DataLoader(vis_train_dataset, batch_size=128, shuffle=True, collate_fn=SingleCollator(tokenizer))
    valid_dataloader = DataLoader(vis_valid_dataset, batch_size=128, shuffle=False,
                                  collate_fn=SingleCollator(tokenizer))

    # Load the BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

    # Set up GPU training if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    epochs = 5
    best_acc = 0
    print('training with {} data'.format(model_name))
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            labels = F.one_hot(labels.to(torch.int64), num_classes=6).float().to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        all_pred = []
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=6).float().to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels_onehot)
                loss = outputs.loss
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.logits, dim=1)

                all_pred.extend(list(predicted.cpu().numpy()))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_dataloader)
        accuracy = correct / total

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Valid Loss: {valid_loss:.4f} - "
              f"Accuracy: {accuracy:.4f}")

        if accuracy > best_acc:
            with open(saving_dir + 'results_{}'.format(model_name), 'a+') as file:
                file.write(f"Epoch {epoch + 1}/{epochs} -\n "
                           f"Train Loss: {train_loss:.4f} -\n "
                           f"Valid Loss: {valid_loss:.4f} -\n "
                           f"Accuracy: {accuracy:.4f}\n")

            best_acc = accuracy
            with open(saving_dir + 'pred_test_best_model_pred_{}'.format(model_name), 'w') as file:
                json.dump(list(map(int, all_pred)), file, indent=4)


if __name__ == "__main__":
    main()
