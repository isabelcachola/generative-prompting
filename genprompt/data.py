'''
Loads and processes each dataset 
'''
import argparse

templates = {
    "TED": {
        "zero": "What is the {} translation of {} A: {}"
        "few": "{} = {}"
    }
}
class MyDataset:
    def __init__(self):
        pass
    
def apply_template(template, context: str, prompt: bool, answer: Optional[str] = '', language: Optional[str] = None):
    applied = None
    if language:
        applied = template.format(language, context, answer)
    else:
        applied = template.format(context, answer)
    if prompt:
        applied = applied[:-1] #remove trailing space
    return applied

def main(context_file, answer_file=None):

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroshot", default=False, action="store_true")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--context_file", type=str, required=True)
    parser.add_argument("--answer_file", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()