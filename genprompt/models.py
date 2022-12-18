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
import logging

logger = logging.getLogger(__name__ +'.models')

class Lead3:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def generate_from_prompts(self, examples, fout_path):
        results = []
        fout = open(fout_path, 'w')
        for ex in tqdm(examples):
            # import ipdb;ipdb.set_trace()
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
    def __init__(self, max_length, batch_size, model_name, device):
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id, max_length=max_length).to(device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = pipeline('text-generation', 
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    batch_size=self.batch_size,
                                    device=0)
        self.max_length = max_length
        self.max_length_generation = 100

    def generate_from_prompts(self, examples: Iterable[str], fout_path:str) -> List[str]:
        fout = open(fout_path, 'w')
        #n = len(examples)
        dataset = Dataset.from_dict(examples)
        # for ex in tqdm(examples, ncols=0):
        # for idx in range(0, n-self.batch_size, self.batch_size):
        for generations in tqdm(self.generator(KeyDataset(dataset, "text"), 
                                        batch_size=self.batch_size, 
                                        max_length=self.max_length-self.max_length_generation, 
                                        pad_token_id=50256, 
                                        clean_up_tokenization_spaces=True,
                                        num_return_sequences=1, 
                                        return_full_text=False)):
            # batch = examples[idx:min(idx+self.batch_size, n)]
            # start_tokenize = time()
            # inputs = self.tokenizer(batch, truncation=True, padding=True, 
            #                         return_tensors="pt", 
            #                         max_length=self.max_length-self.max_length_generation).to(self.device)
            # end_tokenize = time()
            # logger.debug(f'time to tokenize = {end_tokenize - start_tokenize}')

            # outputs = self.model.generate(**inputs, do_sample=True, top_k=50)
            # end_generate = time()
            # logger.debug(f'time to generate = {end_generate - end_tokenize}')

            # generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # logger.debug(f'time to decode = {time() - end_generate}')

            # generations = self.generator(batch, max_length=self.max_length-self.max_length_generation, 
            #                             pad_token_id=50256, 
            #                             num_return_sequences=1, 
            #                             return_full_text=False)
            logging.debug(f'generations={generations}')
            # logger.debug(f'Ouput shape = {outputs.shape}')
            for ex_idx, ex in enumerate(generations):
                # logger.debug(f'input shape, ex_id ={inputs.input_ids.shape}, {ex_idx}' )
                # logger.debug(f'input[exid] shape = {inputs.input_ids.shape}')
                # ex = ex[:,inputs.input_ids[ex_idx].shape[1]-1:] # remove prompts
                fout.write(ex['generated_text'].replace('\n', '') + '\n')
                logger.debug(f'Gen: {repr(ex["generated_text"])}')
            # output += generations[0]['generated_text']
            # output += generation
        fout.close()

    def test_prompt(self) -> List[str]:
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        output = self.generate_from_prompts(examples)
        return output
