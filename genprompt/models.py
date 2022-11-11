'''
Loads and interacts with each model 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model:
    def __init__(self):
        pass
    
class GPTJ:
    def __init__(self):

        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    
    def test_prompt(self):
        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
