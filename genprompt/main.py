'''
Main script
'''
import argparse
import logging
import time

from genprompt import data, models

def init_model(model_name):
    if model_name == 'gptj':
        return models.GPTJ()
    else:
        raise ValueError()

def main(args):
    model = init_model(args.model_name)
    model.test_prompt()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gptj', choices=['gptj'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    # Do things here
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')