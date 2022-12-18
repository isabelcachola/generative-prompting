import sys
import random

# use: 
# python sample.py src tgt number_samples output_prefix

src_file = sys.argv[1]
tgt_file = sys.argv[2]
n = int(sys.argv[3])
outfile = sys.argv[4]
src_lang = src_file.split('.')[-1]
tgt_lang = tgt_file.split('.')[-1]
src_lines = open(src_file).readlines()
tgt_lines = open(tgt_file).readlines()
orig_len = len(src_lines)

print(src_file)
print(orig_len)

if n > orig_len:
  with open(outfile+'.'+src_lang, 'w+') as src_out:
    with open(outfile+'.'+tgt_lang, 'w+') as tgt_out:
      for i in range(orig_len):
        src_out.write(src_lines[i])
        tgt_out.write(tgt_lines[i])
else:
  sample = random.sample(range(orig_len), n)
  with open(outfile+'.'+src_lang, 'w+') as src_out:
    with open(outfile+'.'+tgt_lang, 'w+') as tgt_out:
      for i in sample:
        src_out.write(src_lines[i])
        tgt_out.write(tgt_lines[i])
