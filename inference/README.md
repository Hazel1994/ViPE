You can directly use the model to generate detailed prompts for any arbitrary text.


```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate(text, model, tokenizer,device,do_sample,top_k=100, epsilon_cutoff=.00005, temperature=1):
    #mark the text with special tokens
    text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    #how many new tokens to generate at max
    max_prompt_length=50

    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length, do_sample=do_sample,top_k=top_k, epsilon_cutoff=epsilon_cutoff, temperature=temperature)
    #return only the generated prompts
    pred_caps = tokenizer.batch_decode(generated_ids[:, -(generated_ids.shape[1] - input_ids.shape[1]):], skip_special_tokens=True)

    return pred_caps

device='cpu'
model = GPT2LMHeadModel.from_pretrained('fittar/ViPE-M-CTX7')
model.to(device)

#ViPE-M's tokenizer is identical to that of GPT2-Medium
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# A list of abstract/figurative or any arbitrary combinations of keywords
texts=['lalala', 'I wanna start learning', 'free your mind; you will see the other side of life', 'brave; fantasy']

prompts=generate(texts,model,tokenizer,do_sample=True,device=device)
for t,p in zip(texts,prompts):
    print('{} --> {}'.format(t,p))

lalala -->  A group of people chanting "la la la" around a bonfire on a beach at night
I wanna start learning -->  A child sitting in a library surrounded by books, excitedly flipping through pages of a book
free your mind; you will see the other side of life -->  An astronaut floating in space with a sense of floating weightlessness, looking down towards the earth
brave; fantasy -->  A brave knight with shining armor fighting a fierce dragon in a misty forest

```

**Model Versions:** [ViPE-M-CTX7](https://huggingface.co/fittar/ViPE-M-CTX7) (355M parameters) and [ViPE-S-CTX7](https://huggingface.co/fittar/ViPE-S-CTX7) (117M parameters)
 
### Recommendations

You can use either a comma or a semicolon to combine multiple keywords. for example ['dark, fantasy, brave'] or  ['This is gonna be the best day of my life; do you agree?'].
However, a semicolon draws a stronger boundary between the keywords and encourages the model to transfer the last keyword in a given context (previous keywords).
