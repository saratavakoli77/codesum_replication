"""
This file is to create the sentence bleus of translations (source code to comments).
It uses sentence bleu as opposed to the corpus bleu. It averages the bleu score across
all examples.
"""

import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

delimiters = ";", ",", " ", ".", "\n"
regexPattern = '|'.join(map(re.escape, delimiters))

def get_unigram_dict(corpus_file_path):
    unigram_dict_code = {}
    with open(corpus_file_path) as f1:
        line = f1.readline()
        code_token_count = 0
        while line:
            line = re.split(regexPattern, line)
            for token in line:
                code_token_count+=1
                if token == "\n":
                    break
                if token == "<s>" or token == "</s>" or token == "":
                    continue
                if token not in unigram_dict_code:
                    unigram_dict_code[token] = 1
                else:
                    unigram_dict_code[token]+=1
            line = f1.readline()

        sorted_list_code = sorted(unigram_dict_code.items(), key=lambda item: item[1], reverse=True)
        return sorted_list_code

def remove_tokens_from_line(input_line, list_of_removable_tokens):
    input_line_split = input_line.split()
    new_input_line = []
    for token in input_line_split:
        if token in list_of_removable_tokens:
            new_input_line += ["<PAD_TOKEN>"]
        else:
            new_input_line += [token]
    return new_input_line

def calculate_bleu_scores(translation_file_path, translation_ref_file_path):
    in_f1 = open(translation_file_path, "r")
    in_f2 = open(translation_ref_file_path, "r")

    line1 = in_f1.readline()
    line2 = in_f2.readline()

    preds = []
    refs = []

    while line1:
        preds.append(line1.split())
        refs.append([line2.split()])
        line1 = in_f1.readline()
        line2 = in_f2.readline()

    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    print(Ba)
    print(B1)
    print(B2)
    print(B3)
    print(B4)


if __name__ == "__main__":
    calculate_bleu_scores("reference_translations/coms-translation.test", "reference_translations/coms-cleaned.test")





