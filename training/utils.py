import pandas as pd
from torch.utils.data import Dataset, DataLoader


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Dataset(Dataset):

    def __init__(self, data_dir, context_size, training=True):

        data=pd.read_csv(data_dir)
        data = data.sample(frac=1,random_state=0).reset_index(drop=True)

        self.context_size = context_size
        self.ids_2_sample={}
        self.keys=[str(i)+':'+str(j) for i,j in zip(data['ids'],data['gpt_ids'])]
        values = zip(data['lyrics'], data['prompts'])
        #self.titles= [i+ ' by '  + j for i,j in  zip(data['titles'],data['artists'])]

        #its the same vali and train, but keys are gonna be different
        self.ids_2_sample = {k: v for k, v in zip(self.keys, values)}
        valid_index = int(.10 * len(self.keys))

        if training:
            self.keys = self.keys[valid_index:]
            #self.titles = self.titles[valid_index:]
        else:
            self.keys = self.keys[0:valid_index]
            #self.titles = self.titles[0:valid_index]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key= self.keys[idx]
        context,prompt = self.ids_2_sample[key]
        context = str(context) # a couple of nan cases exist but that should not matter give the size of the dataset

        #extend the context by context size if possible
        if self.context_size > 0:
            for c in range(self.context_size,0,-1):
                key_id, key_gpt_id=key.split(':')
                key_id=int(key_id)
                key_id -= c
                potential_key='{}:{}'.format(key_id,key_gpt_id)

                if potential_key in self.ids_2_sample:
                    context = str(self.ids_2_sample[potential_key][0]) + ' ; ' + context

        return context, prompt

class ContextAwareDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token=self.tokenizer.eos_token

    def collator(self, batch):
        prompts = []
        contexts = []
        for context, prompt in batch:
            prompts.append( self.eos_token + prompt + self.eos_token )
            contexts.append(self.eos_token + context )

        tokens=self.tokenizer(contexts,prompts , padding=True, return_token_type_ids=True, return_tensors="pt")

        return tokens

    def __call__(self, batch):
       return self.collator(batch)


#
# def generate_from_loader(valid_gen, model, tokenizer,device,do_sample):
#     transformers.utils.logging.set_verbosity_error()
#     id2cap={}
#     ground_truth={}
#
#     if do_sample> 0:
#         do_sample=True
#     else:
#         do_sample= False
#
#     all_ids=[]
#     all_generated_ids=[]
#     all_prompts=[]
#
#     # Set token type IDs for the prompts
#     max_prompt_length = 30
#
#
#     for bc,batch in enumerate(tqdm(valid_gen)):
#
#
#         tokens=batch['tokens']
#         # Generate text
#         with torch.no_grad():
#             input_ids = tokens["input_ids"].to(device)
#             attention_mask = tokens["attention_mask"].to(device)
#             max_length = input_ids.shape[1] + max_prompt_length
#             generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length,do_sample=do_sample)
#
#         all_generated_ids.append(generated_ids)
#         all_ids.append( batch['keys'])
#         all_prompts.append(batch['prompts'])
#
#     print('processing the generated captions')
#     for bc, ids, in enumerate(tqdm(all_ids)):
#
#         pred_caps = tokenizer.batch_decode(all_generated_ids[bc][:,-max_prompt_length:], skip_special_tokens=True)
#         #input_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         for c, (id, pred_cap) in enumerate(zip(ids,pred_caps)):
#             generated_part = pred_cap[1:]
#             id2cap[id] = generated_part
#             ground_truth[id] = all_prompts[bc][c][1:]
#
#             # print(input_prompts[c])
#             # print(generated_part)
#             # print( ground_truth[id])
#             # print('\n')
#
#     return id2cap,ground_truth
#
# def generate_from_sentences(text, model, tokenizer,device,do_sample,top_k, num_beams,epsilon_cutoff=.0005, temperature=1.1):
#     text=[tokenizer.eos_token +  i  + tokenizer.eos_token for  i in text]
#     #start=[ tokenizer.eos_token for _ in text]
#     batch=tokenizer(text, padding=True, return_tensors="pt")
#
#     input_ids = batch["input_ids"].to(device)
#     attention_mask = batch["attention_mask"].to(device)
#     #token_type_ids = batch['token_type_ids'].to(device)
#
#     # Set token type IDs for the prompts
#     max_prompt_length=35
#     # token_type_ids[attention_mask == 0]=1
#     #prompts_token_type_ids = torch.ones(token_type_ids.shape[0],max_prompt_length, dtype=torch.long).to(device)
#     ##Extend token_type_ids to cover the prompt segment
#     #token_type_ids = torch.cat((token_type_ids, prompts_token_type_ids),dim=-1)
#
#     # token_type_ids = torch.tensor(
#     #     [[0] * input_ids.shape[1] + [1] * max_prompt_length for _ in range(input_ids.shape[0])]).to(device)
#     # labels = input_ids.clone()
#     #pred_caps_1=gen(model, batch,tokenizer)
#     max_length=input_ids.shape[1] + max_prompt_length
#     if num_beams>0:
#         generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,
#                                        do_sample=False, num_beams=num_beams)
#     elif top_k >0:
#         generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_length=max_length, do_sample=do_sample,top_k=top_k,epsilon_cutoff=epsilon_cutoff, temperature=temperature)
#     else:
#         generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,
#                                        do_sample=do_sample,epsilon_cutoff=epsilon_cutoff, temperature=temperature)
#
#     #generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
#                                   # token_type_ids=token_type_ids, max_length=max_length)
#     pred_caps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#
#     for prompt in pred_caps:
#         print(prompt)
#
#     #return pred_caps
#
# class DatasetTest(Dataset):
#
#     def __init__(self, data_dir, context_size, training=True):
#
#         data=pd.read_csv(data_dir)
#         data = data.sample(frac=1,random_state=0).reset_index(drop=True)
#
#         self.context_size = context_size
#         self.ids_2_sample={}
#         self.keys=[str(i)+':'+str(j) for i,j in zip(data['ids'],data['gpt_ids'])]
#         values = zip(data['lyrics'], data['prompts'])
#         #self.titles= [i+ ' by '  + j for i,j in  zip(data['titles'],data['artists'])]
#
#         #its the same vali and train, but keys are gonna be different
#         self.ids_2_sample = {k: v for k, v in zip(self.keys, values)}
#         valid_index = int(.10 * len(self.keys))
#
#         if training:
#             self.keys = self.keys[valid_index:]
#             #self.titles = self.titles[valid_index:]
#         else:
#             self.keys = self.keys[0:valid_index]
#             #self.titles = self.titles[0:valid_index]
#
#
#     def __len__(self):
#         return len(self.keys)
#
#     def __getitem__(self, idx):
#
#         key= self.keys[idx]
#         context,prompt = self.ids_2_sample[key]
#         context = str(context) # a couple of nan cases exist but that should not matter give the size of the dataset
#
#         #extend the context by context size
#         if self.context_size > 0:
#             for c in range(self.context_size,0,-1):
#                 key_id, key_gpt_id=key.split(':')
#                 key_id=int(key_id)
#                 key_id -= c
#                 potential_key='{}:{}'.format(key_id,key_gpt_id)
#
#                 if potential_key in self.ids_2_sample:
#                     context = str(self.ids_2_sample[potential_key][0]) + ' ; ' + context
#
#         return context, prompt,int(key.split(':')[0])
#
#
# class ContextAwareDataCollatorForGeneration:
#
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.eos_token=self.tokenizer.eos_token
#
#     def collator(self, batch):
#         prompts = []
#         keys=[]
#         contexts = []
#         for context, prompt,key in batch:
#             keys.append( key )
#             contexts.append(self.eos_token + context + self.eos_token)
#             prompts.append(prompt)
#
#         tokens=self.tokenizer(contexts,padding=True, return_token_type_ids=False, return_tensors="pt")
#         return {'tokens':tokens,'contexts':contexts,'prompts':prompts,'keys':keys}
#
#     def __call__(self, batch):
#        return self.collator(batch)
#
# def to_coco_format(test_annotations, output_json):
#
#     annFile=[]
#     id2captions = {}
#     for id, cap in test_annotations.items():
#         annFile.append({'image_id': id, 'sentences': [cap]})
#
#     annot_format = {
#         'info': {
#             'year': 2014,
#             'version': '1.0',
#             'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.',
#             'contributor': 'Microsoft COCO group',
#             'url': 'http://mscoco.org',
#             'date_created': '2015-01-27 09:11:52.357475'
#         },
#         'licenses': [
#             {
#                 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
#                 'id': 1,
#                 'name': 'Attribution-NonCommercial-ShareAlike License'
#             },
#             {
#                 'url': 'http://creativecommons.org/licenses/by-nc/2.0/',
#                 'id': 2,
#                 'name': 'Attribution-NonCommercial License'
#             },
#             {
#                 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
#                 'id': 3,
#                 'name': 'Attribution-NonCommercial-NoDerivs License'
#             },
#             {
#                 'url': 'http://creativecommons.org/licenses/by/2.0/',
#                 'id': 4,
#                 'name': 'Attribution License'
#             },
#             {
#                 'url': 'http://creativecommons.org/licenses/by-sa/2.0/',
#                 'id': 5,
#                 'name': 'Attribution-ShareAlike License'
#             },
#             {
#                 'url': 'http://creativecommons.org/licenses/by-nd/2.0/',
#                 'id': 6,
#                 'name': 'Attribution-NoDerivs License'
#             },
#             {
#                 'url': 'http://flickr.com/commons/usage/',
#                 'id': 7,
#                 'name': 'No known copyright restrictions'
#             },
#             {
#                 'url': 'http://www.usa.gov/copyright.shtml',
#                 'id': 8,
#                 'name': 'United States Government Work'
#             }
#         ],
#         'type': 'captions',
#         'images': [],
#         'annotations': []
#     }
#
#     count = 0
#     for annot in tqdm(annFile):
#         annot_format['images'].append({
#             "id": annot.get("cocoid", annot['image_id']),
#             "width": 512,
#             "height": 512,
#             "filename": "000" + str(annot['image_id']),
#             "license": 1,
#             "flickr_url": '',
#             "coco_url": '',
#             "date_captured": ''
#         })
#
#         for sent in annot['sentences']:
#             annot_format['annotations'].append({
#                 "id": count,
#                 "image_id": annot.get("cocoid", annot['image_id']),
#                 "caption": sent
#             })
#             count += 1
#
#     with open(output_json, 'w') as f:
#         json.dump(annot_format, f)
#
# def load_pkl(file):
#     with open(file, 'rb') as handle:
#         return pickle.load(handle)
#
# def save_pkl(obj, filename):
#
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
