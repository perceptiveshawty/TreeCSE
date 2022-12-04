import os
import argparse
from datasets import load_dataset

ds = load_dataset('wikitext', 'wikitext-103-raw-v1')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./wikitext103')
    parser.add_argument('--min_char_count', type=int, default=400)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    threshold = args.min_char_count

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'passages.txt'), 'a') as t:
        for ex in ds['train']:
            if len(ex['text']) > threshold:
                t.write(str(ex['text'].strip()) + '\n')