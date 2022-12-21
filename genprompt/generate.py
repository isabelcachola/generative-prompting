from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm 
import json
import argparse
import time

MAX_NEW_TOKENS=80

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", 
                    truncation_side="left", truncation="max_length", max_length=2048-MAX_NEW_TOKENS)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if args.model == "EleutherAI/gpt-neo-1.3B":
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision="float16", torch_dtype=torch.float16).to(device)

    lines = None
    with open(args.input_file, encoding='utf-8') as f:
        lines = json.load(f)

    lines_length = len(lines)
    line_list = list(lines.values())
    batch_sz = args.batch
    i = 0
    with torch.no_grad():
        fout = open(args.output_file, 'w')
        pbar = tqdm(total=lines_length, ncols=0, disable=not args.verbose)
        while i < lines_length:
            prompt_batch = line_list[i:i+batch_sz]
            toked = tokenizer(prompt_batch, return_tensors='pt', padding=True, truncation=True, max_length=2048-MAX_NEW_TOKENS).to(device)
            tokenizer.pad_token = tokenizer.eos_token
            
            gen_tokens = model.generate(
                **toked,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            gen_text = tokenizer.batch_decode(gen_tokens[:,toked.input_ids.shape[1]:], skip_special_tokens=True)
            for line in gen_text:
                fout.write(line.replace('\n', ' ').strip() + '\n')
                # print(line.split('\n')[0].strip())

            i+=batch_sz
            pbar.update(batch_sz)
        fout.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()
    start = time.time()
    main(args)
    end = time.time()
    print(f'Time to run script = {(end-start)/60} mins')

