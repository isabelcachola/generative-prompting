'''
Loads and interacts with each model 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers import GPTJForCausalLM
from transformers import pipeline
from typing import List, Iterable
from transformers.pipelines.pt_utils import KeyDataset
import spacy
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from time import time
import torch
import logging
import os
import openai

logger = logging.getLogger(__name__ +'.models')
logging.getLogger("openai").setLevel(logging.WARNING)

class Lead3:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def generate_from_prompts(self, examples, fout_path):
        results = []
        fout = open(fout_path, 'w')
        for ex in tqdm(examples['text']):
            this_ex = []
            text = ex.split('\n')[0]
            tokens = self.nlp(text)
            for i, s in enumerate(tokens.sents):
                this_ex.append(s.text)
                if i==2:
                    break
            results.append(' '.join(this_ex))
            fout.write(' '.join(this_ex) + '\n')
        fout.close()
        return results

    
class GPT:
    def __init__(self, max_length, model_name, device):
        self.batch_size = 2
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                            # revision="float16", 
                                                            # torch_dtype=torch.float16
                                                            ).to(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.max_length_generation = 100

    def generate_from_prompts(self, examples: Iterable[str], fout_path:str) -> List[str]:
        fout = open(fout_path, 'w')
        lines_length = len(examples)
        i = 0
        with torch.no_grad():
            pbar = tqdm(total=lines_length)
            while i < lines_length:
                prompt_batch = examples[i:i+self.batch_size]
                toked = self.tokenizer(prompt_batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                gen_tokens = self.model.generate(
                    **toked,
                    do_sample=False,
                    max_new_tokens=20,
                )
                gen_text = self.tokenizer.batch_decode(gen_tokens[:,toked.input_ids.shape[1]:], skip_special_tokens=True)
                for line in gen_text:
                    print(line)
                    fout.write(line + '\n')
                pbar.update(self.batch_size)
                i += self.batch_size
            pbar.close()
        fout.close()

    def test_prompt(self) -> List[str]:
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        output = self.generate_from_prompts(examples)
        return output

class GPT3:
    def __init__(self, model_name, batch_size=32):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.batch_size = batch_size    
        
    def get_response(self, prompt, temperature=0):
        response = openai.Completion.create(model=self.model_name, 
                                            prompt=prompt, 
                                            temperature=temperature,
                                            max_tokens=100
                                            )
        return response

    def format_response(self, response):
        text = response['text'].replace('\n', ' ').strip()
        return text

    def generate_from_prompts(self, examples: Iterable[str], fout_path:str) -> List[str]:
        fout = open(fout_path, 'w')
        lines_length = len(examples)
        logger.info(f'Num examples = {lines_length}')
        i = 0
        for i in tqdm(range(0, lines_length, self.batch_size), ncols=0):
            prompt_batch = examples[i:min(i+self.batch_size, lines_length)]
            try:
                response = self.get_response(prompt_batch)
                for line in response['choices']:
                    line = self.format_response(line)
                    fout.write(line + '\n')
            except:
                for i in range(len(prompt_batch)):
                    try:
                        _r = self.get_response(prompt_batch[i])['choices'][0]
                        line = self.format_response(_r)
                        fout.write(line + '\n')
                    except:
                        l_prompt = len(prompt_batch[i])
                        _r = self.get_response(prompt_batch[i][l_prompt-2000:])['choices'][0]
                        line = self.format_response(_r)
                        fout.write(line + '\n')

        fout.close()

