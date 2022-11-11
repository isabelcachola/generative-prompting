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
    def __init__(self, max_length, model_name):
        self.generator = pipeline('text-generation', model=model_name)
        self.max_length = max_length

    def generate_from_prompts(self, examples: List[str]) -> List[str]:
        n_examples = len(examples)
        output = []
        for ex in examples:
            generations = self.generator(ex, max_length=self.max_length, pad_token_id=50256, 
                                        num_return_sequences=1, return_full_text=False, 
                                        handle_long_generation="hole")
            output += generations[0]['generated_text']
        return output

    def test_prompt(self) -> List[str]:
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        output = self.generate_from_prompts(examples)
        return output
