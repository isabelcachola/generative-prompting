'''
Loads and interacts with each model 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers import GPTJForCausalLM
from transformers import pipeline
from typing import List
import spacy
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

class Lead3:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def generate_from_prompts(self, examples):
        results = []
        for ex in examples:
            this_ex = []
            text = ex.split('\n')[0]
            tokens = self.nlp(text)
            for i, s in enumerate(tokens.sents):
                this_ex.append(s.text)
                if i==2:
                    break
            results.append(' '.join(this_ex))
        return results

    
class GPT:
    def __init__(self, max_length, model_name, device):
        # self.generator = pipeline('text-generation', model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id, max_length=max_length).to(device)
        self.device = device
        self.max_length = max_length
        self.max_length_generation = 100

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        n_examples = len(examples)
        output = []
        for ex in tqdm(examples):
            inputs = self.tokenizer(ex, truncation=True, return_tensors="pt", max_length=self.max_length-self.max_length_generation).to(self.device)
            outputs = self.model.generate(**inputs, do_sample=True, top_k=50)
            outputs = outputs[:,inputs.input_ids.shape[1]-1:] # remove prompts
            # inputs.input_ids.sha
            # outputs.shape
            generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # import ipdb;ipdb.set_trace()
            # generations = self.generator(ex, max_length=self.max_length, pad_token_id=50256, 
            #                             num_return_sequences=1, return_full_text=False)
            # output += generations[0]['generated_text']
            output += generation
        return output

    def test_prompt(self) -> List[str]:
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        output = self.generate_from_prompts(examples)
        return output
