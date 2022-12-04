import os
import ndjson
import argparse
from RST import RST


def parse_args():
    parser = argparse.ArgumentParser(description='RST Dataset Curation')
    parser.add_argument('--strategy', type=str, default='deep', choices=['full', 'deep', 'shallow'],
                        help='whether to sample from the whole (full) RST, or bottom/top (deep/shallow) of the RST')
    parser.add_argument('--data_dir', type=str, default='./data/wikitext103',
                        help='path to data dir; expects output of DMRST_Parser + original text file (4 total files)')
    return parser.parse_args()

def merge_edus(items, k=3): # auxiliary function to combine adjacent shortest edus until list is length k
    assert len(items)
    if len(items) == 2:
        return items + [items[0] + " " + items[1]]
    elif len(items) == 1:
        return [items[0]]*3
    while len(items) != k:
        index = items.index(min(items, key=len))
        if index == 0: # merge right
            merged = " ".join(items[:2]) # 0, 1 -> 0
            items = [merged] + items[2:] 
        elif index == len(items) - 1: # merge left
            merged = " ".join(items[-2:]) # -2, -1 -> -1
            items = items[:-2] + [merged]
        elif len(items[index - 1]) > len(items[index + 1]): # merge right
            if index == len(items) - 2: # if index == -2
                merged = " ".join(items[index:]) # -2, -1 -> -1
                items = items[:index] + [merged]
            else:
                merged = " ".join(items[index:index+2])
                items = items[:index] + [merged] + items[index+2:]
        else: # merge left
            if index == 1:
                merged = " ".join(items[:2]) # 0, 1 -> 0
                items = [merged] + items[2:]
            else:
                merged = " ".join(items[index-1:index+1])
                items = items[:index-1] + [merged] + items[index+1:]
    return items

def make_full_rst_dataset(output_dir):
    examples = []
    with open(os.path.join(output_dir, 'passages.txt'), 'r') as p_, open(os.path.join(output_dir, 'tokenization.txt'), 'r') as t_, open(os.path.join(output_dir, 'segmentation.txt'), 'r') as e_, open(os.path.join(output_dir, 'tree.txt'), 'r') as s_:
        orig_examples, rst_tokens, rst_edus, rst_rels = p_.readlines(), t_.readlines(), e_.readlines(), s_.readlines()
    
    for index, ex in enumerate(orig_examples):
        try:
            tokenization = rst_tokens[index].strip()[1:-1].split(", ")
            tokenization = [t[1:-1] for t in tokenization]
            segmentation = [int(k) for k in rst_edus[index].strip()[1:-1].split(", ")]
            parsetree = rst_rels[index].strip()[2:-2].split()
            rst_example = RST.from_data(tokenization, segmentation, parsetree)
        except:
            continue
        
        all_left_args, all_right_args, all_schemes = RST.all_rst_cutouts(rst_example)
        all_cutouts_valid = [(sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]), left_, right_, scheme_) for left_, right_, scheme_ in zip(all_left_args, all_right_args, all_schemes) if sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]) > 20]
        _, left_arg_edus, right_arg_edus, _ = zip(*sorted(all_cutouts_valid, key=lambda x:x[0], reverse=True))

        for lae, rae in zip(left_arg_edus, right_arg_edus):
            all_edus = [e.text for e in lae] + [e.text for e in rae]
            anchor = " ".join(all_edus)
            p1, p2, p3 = merge_edus(all_edus)
            examples.append({'anchor' : anchor, 'pos1' : p1, 'pos2' : p2, 'pos3' : p3})

    with open(os.path.join(output_dir, 'composition_data_for_cse.json'), 'w') as jout:
        ndjson.dump(examples, jout)

def make_shallow_rst_dataset(output_dir):
    examples = []
    with open(os.path.join(output_dir, 'passages.txt'), 'r') as p_, open(os.path.join(output_dir, 'tokenization.txt'), 'r') as t_, open(os.path.join(output_dir, 'segmentation.txt'), 'r') as e_, open(os.path.join(output_dir, 'tree.txt'), 'r') as s_:
        orig_examples, rst_tokens, rst_edus, rst_rels = p_.readlines(), t_.readlines(), e_.readlines(), s_.readlines()
    
    for index, ex in enumerate(orig_examples):
        try:
            tokenization = rst_tokens[index].strip()[1:-1].split(", ")
            tokenization = [t[1:-1] for t in tokenization]
            segmentation = [int(k) for k in rst_edus[index].strip()[1:-1].split(", ")]
            parsetree = rst_rels[index].strip()[2:-2].split()
            rst_example = RST.from_data(tokenization, segmentation, parsetree)
        except:
            continue

        all_left_args, all_right_args, all_schemes = RST.all_rst_cutouts(rst_example)
        all_cutouts_valid = [(sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]), left_, right_, scheme_) for left_, right_, scheme_ in zip(all_left_args, all_right_args, all_schemes) if sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]) > 40]
        _, left_arg_edus, right_arg_edus, _ = zip(*sorted(all_cutouts_valid, key=lambda x:x[0], reverse=True))

        deep = 0
        for lae, rae in zip(left_arg_edus, right_arg_edus):
            if deep <= len(left_arg_edus) // 2:
                all_edus = [e.text for e in lae] + [e.text for e in rae]
                anchor = " ".join(all_edus)
                p1, p2, p3 = merge_edus(all_edus)
                examples.append({'anchor' : anchor, 'pos1' : p1, 'pos2' : p2, 'pos3' : p3})
            deep += 1

    with open(os.path.join(output_dir, 'shallow_data_for_cse.json'), 'w') as jout:
        ndjson.dump(examples, jout)

def make_deep_rst_dataset(output_dir):
    examples = []
    with open(os.path.join(output_dir, 'passages.txt'), 'r') as p_, open(os.path.join(output_dir, 'tokenization.txt'), 'r') as t_, open(os.path.join(output_dir, 'segmentation.txt'), 'r') as e_, open(os.path.join(output_dir, 'tree.txt'), 'r') as s_:
        orig_examples, rst_tokens, rst_edus, rst_rels = p_.readlines(), t_.readlines(), e_.readlines(), s_.readlines()
    
    for index, ex in enumerate(orig_examples):
        try:
            tokenization = rst_tokens[index].strip()[1:-1].split(", ")
            tokenization = [t[1:-1] for t in tokenization]
            segmentation = [int(k) for k in rst_edus[index].strip()[1:-1].split(", ")]
            parsetree = rst_rels[index].strip()[2:-2].split()
            rst_example = RST.from_data(tokenization, segmentation, parsetree)
        except:
            continue

        all_left_args, all_right_args, all_schemes = RST.all_rst_cutouts(rst_example)
        all_cutouts_valid = [(sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]), left_, right_, scheme_) for left_, right_, scheme_ in zip(all_left_args, all_right_args, all_schemes) if sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]) > 40]
        _, left_arg_edus, right_arg_edus, _ = zip(*sorted(all_cutouts_valid, key=lambda x:x[0], reverse=False))

        deep = 0
        for lae, rae in zip(left_arg_edus, right_arg_edus):
            if deep <= len(left_arg_edus) // 2:
                all_edus = [e.text for e in lae] + [e.text for e in rae]
                anchor = " ".join(all_edus)
                p1, p2, p3 = merge_edus(all_edus)
                examples.append({'anchor' : anchor, 'pos1' : p1, 'pos2' : p2, 'pos3' : p3})
            deep += 1

    with open(os.path.join(output_dir, 'deep_data_for_cse.json'), 'w') as jout:
        ndjson.dump(examples, jout)

if __name__ == '__main__':
    args = parse_args()
    if args.strategy == 'full':
        make_full_rst_dataset(args.data_dir)
    elif args.strategy == 'shallow':
        make_shallow_rst_dataset(args.data_dir)
    else:
        make_deep_rst_dataset(args.data_dir)