import os
import ndjson
import argparse
from RST import RST

def parse_args():
    parser = argparse.ArgumentParser(description='RST Dataset Curation')
    parser.add_argument('--strategy', type=str, default='tree', choices=['tree', 'edu'],
                        help='whether to create the dataset to train the teacher or student')
    parser.add_argument('--data_dir', type=str, default='./data/wikitext103',
                        help='path to data dir; expects output of DMRST_Parser + original text file (4 total files)')
    return parser.parse_args()

def merge_edus(items, k=3): # auxiliary function to combine adjacent shortest edus until list is length k
    assert len(items)
    if k == 3:
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
    else:
        if len(items) == 1:
            raise Exception

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


def make_rst_dataset(output_dir, strategy):
    examples = []
    label2int = {}
    int2freq = {}
    intlabel = 0
    path_to_sentences = 'wiki1m_for_simcse.txt' if 'wiki1m' in output_dir else 'passages.txt'
    with open(os.path.join(output_dir, path_to_sentences), 'r') as p_, open(os.path.join(output_dir, 'tokenization.txt'), 'r') as t_, open(os.path.join(output_dir, 'segmentation.txt'), 'r') as e_, open(os.path.join(output_dir, 'tree.txt'), 'r') as s_:
        orig_examples, rst_tokens, rst_edus, rst_rels = p_.readlines(), t_.readlines(), e_.readlines(), s_.readlines()
    
    num_examples = len(rst_rels)
    for index in range(num_examples):
        try:
            tokenization = rst_tokens[index].strip()[1:-1].split(", ")
            tokenization = [t[1:-1] for t in tokenization]
            segmentation = [int(k) for k in rst_edus[index].strip()[1:-1].split(", ")]
            parsetree = rst_rels[index].strip()[2:-2].split()
            rst_example = RST.from_data(tokenization, segmentation, parsetree)
        except:
            # tokens = ex.split()
            # word_count = len(tokens)
            # parent, left, right = ex, " ".join(tokens[:word_count]).strip(), " ".join(tokens[word_count:]).strip()
            # examples.append({'parent' : parent, 'left' : left, 'right' : right})
            continue

        if strategy == "tree":
            if 'wiki1m' not in output_dir:
                all_left_args, all_right_args, all_schemes = RST.all_rst_cutouts(rst_example)
                all_cutouts_valid = [(sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]), left_, right_, scheme_) for left_, right_, scheme_ in zip(all_left_args, all_right_args, all_schemes) if sum([sum(len(edu__.text) for edu__ in left_), sum(len(edu__.text) for edu__ in right_)]) > 20]
                _, left_arg_edus, right_arg_edus, all_schemes = zip(*sorted(all_cutouts_valid, key=lambda x:x[0], reverse=False))

                deep = 0
                cutoff = len(left_arg_edus) // 3
                for k in range(len(all_schemes)):
                    lae, rae, sch = left_arg_edus[k], right_arg_edus[k], all_schemes[k].split("-")[-1]
                    if deep <= cutoff:
                        left = " ".join([e.text for e in lae])
                        right = " ".join([e.text for e in rae])
                        parent = left + " " + right
                        if sch not in label2int:
                            label2int[sch] = intlabel
                            intlabel += 1
                            int2freq[label2int[sch]] = 0
                        int2freq[label2int[sch]] += 1
                        # all_edus = [e.text for e in lae]
                        # all_edus = all_edus + [e.text for e in rae]
                        # parent = " ".join(all_edus)
                        # left, right =
                        # left, right = merge_edus([e.text for e in lae] + [e.text for e in rae], k=2)
                        examples.append({'parent' : parent, 'left' : left, 'right' : right, 'label' : label2int[sch]})
                    deep += 1
            else:
                left_arg_edus, right_arg_edus, all_schemes = RST.all_rst_cutouts(rst_example)

                for k in range(len(all_schemes)):
                    lae, rae, sch = left_arg_edus[k], right_arg_edus[k], all_schemes[k].split("-")[-1]
                    left = " ".join([e.text for e in lae])
                    right = " ".join([e.text for e in rae])
                    parent = left + " " + right 
                    if sch not in label2int:
                        label2int[sch] = intlabel
                        intlabel += 1
                        int2freq[label2int[sch]] = 0
                    int2freq[label2int[sch]] += 1
                    # all_edus = [e.text for e in lae] + [e.text for e in rae]
                    # parent = " ".join(all_edus)
                    # left, right = " ".join([e.text for e in lae]), " ".join([e.text for e in rae])
                    # left, right = merge_edus(all_edus, k=2)
                    examples.append({'parent' : parent, 'left' : left, 'right' : right, 'label' : label2int[sch]})

        else:
            left_edus, right_edus = RST.get_left_subtree_edus(rst_example), RST.get_right_subtree_edus(rst_example)
            for lae in left_edus:
                examples.append({"edu" : lae})
            for rae in right_edus:
                examples.append({"edu" : rae})

    print("num examples: ", len(examples))
    print("num labels: ", len(label2int))
    print("label2int: ", label2int)
    print("int2freq: ", int2freq)

    weights = [0]*len(label2int)
    for it in range(len(label2int)):
        weights[it] = len(examples) / (int2freq[it] * len(label2int))

    print("class weights: \n", weights)

    save_path = os.path.join(output_dir, 'rst_data_for_cse_v1.json')
    with open(save_path, 'w') as jout:
        ndjson.dump(examples, jout)

if __name__ == '__main__':

    args = parse_args()
    make_rst_dataset(args.data_dir, args.strategy)
    # if args.strategy == 'full':
    #     make_full_rst_dataset(args.data_dir)
    # elif args.strategy == 'shallow':
    #     make_shallow_rst_dataset(args.data_dir)
    # elif args.strategy == 'deep':
    #     make_deep_rst_dataset(args.data_dir)
    # else:
    #     make_simcse_rst_dataset(args.data_dir)