import jsonlines
import os.path as osp
import os
import json
from string_distance.edit_distance import levenshtein

class Ev:
    def __init__(self) -> None:
        self.max_length = 0
        self.max_distance = 0

        self.max_seq = 0
        self.max_right_seq = 0

    def count(self, x, y):
        cost = levenshtein(x, y)
        self.max_distance += cost
        self.max_length += max(len(x), len(y))
        if x == y:
            self.max_right_seq += 1
        self.max_seq += 1
        return  

    def socre(self, ):
        char_acc = 1 - self.max_distance / self.max_length
        seq_acc = self.max_right_seq / self.max_seq
        return dict(char_acc=char_acc, seq_acc=seq_acc, max_seq=self.max_seq, max_right_seq=self.max_right_seq, max_distance=self.max_distance, max_length=self.max_length)


if __name__ == "__main__":
    reader = list(jsonlines.open("result_val.jsonl", "r").iter())
    new_reader = {}
    for d in reader:
        new_reader[d['name']] = d['decode_result']


    with open("../table_ocr/data/val.txt", "r") as f:
        data = f.readlines()
    gt_dict = {}
    for line in data:
        name,tokens = line.strip('\n').split('\t')
        gt_dict[name] = tokens


    ev = Ev()
    for key, tokens in new_reader.items():
        gt_tokens = gt_dict.get(key, False)
        if gt_tokens:
            ev.count(tokens, gt_tokens)

    print(ev.socre())
