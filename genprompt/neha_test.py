from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm 
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16).to(device)

lines = None
with open('out_k1/baseline+ted_dev_en-de.raw.de.json') as f:
    lines = json.load(f)


lines_length = len(lines)
line_list = list(lines.values())
batch_sz = 8
i = 0
with torch.no_grad():
    while i < lines_length:
        prompt_batch = line_list[i:i+batch_sz]
        toked = tokenizer(prompt_batch, return_tensors='pt', padding=True, truncation=True).to(device)
        tokenizer.pad_token = tokenizer.eos_token
        gen_tokens = model.generate(
            **toked,
            do_sample=False,
            max_new_tokens=100,
        )

        gen_text = tokenizer.batch_decode(gen_tokens[:,toked.input_ids.shape[1]:], skip_special_tokens=True)
        for line in gen_text:
            print(line.split('\n')[0].strip())
        i+=batch_sz