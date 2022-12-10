"""
This file is to create the sentence bleu plots of translations (source code to comments)
with the most common 100 unigrams/bigrams/trigrams/tetragrams removes. Create plots are located in 
the plots directory.
"""

import re
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

delimiters = ";", ",", " ", ".", "\n"
regexPattern = '|'.join(map(re.escape, delimiters))

def get_unigram_list(corpus_file_path):
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

def get_bigram_list(corpus_file_path):
    bigram_dict_code = {}
    with open(corpus_file_path) as f_1:
        line = f_1.readline()
        code_bigram_count = 0
        while line:
            line = re.split(regexPattern, line)
            prev_token = ""
            for token in line:
                if token == "\n":
                    break
                if prev_token == "<s>" or token == "</s>" or prev_token == "" or token == "":
                    prev_token = token
                    continue
                code_bigram_count+=1
                bigram = "{}-{}".format(prev_token, token)
                if bigram not in bigram_dict_code:
                    bigram_dict_code[bigram] = 1
                else:
                    bigram_dict_code[bigram]+=1
                prev_token = token
            line = f_1.readline()

    sorted_list_code = sorted(bigram_dict_code.items(), key=lambda item: item[1], reverse=True)
    return sorted_list_code

def get_trigram_list(corpus_file_path):
    trigram_dict_code = {}

    with open(corpus_file_path) as f_1:
        line = f_1.readline()
        code_trigram_count = 0
        while line:
            line = re.split(regexPattern, line)
            prev_token = ""
            prev_prev_token = ""
            for token in line:
                if token == "\n":
                    break
                if prev_prev_token == "<s>" or prev_token == "<s>" or token == "</s>" or prev_prev_token == "" or prev_token == "" or token == "":
                    prev_prev_token = prev_token
                    prev_token = token
                    continue
                code_trigram_count+=1
                trigram = "{}-{}-{}".format(prev_prev_token, prev_token, token)
                if trigram not in trigram_dict_code:
                    trigram_dict_code[trigram] = 1
                else:
                    trigram_dict_code[trigram]+=1
                prev_prev_token = prev_token
                prev_token = token
            line = f_1.readline()
    
    sorted_list_code = sorted(trigram_dict_code.items(), key=lambda item: item[1], reverse=True)
    return sorted_list_code

def get_tetragram_list(corpus_file_path):
    tetragram_dict_code = {}

    with open(corpus_file_path) as f_1:
        line = f_1.readline()
        code_tetragram_count = 0
        while line:
            line = re.split(regexPattern, line)
            prev_token = ""
            prev_prev_token = ""
            prev_prev_prev_token = ""
            for token in line:
                if token == "\n":
                    break
                if prev_prev_prev_token == "<s>" or prev_prev_token == "<s>" or prev_token == "<s>" or token == "</s>" or prev_prev_prev_token == "" or prev_prev_token == "" or prev_token == "" or token == "":
                    prev_prev_prev_token = prev_prev_token
                    prev_prev_token = prev_token
                    prev_token = token
                    continue
                code_tetragram_count+=1
                tetragram = "{}-{}-{}-{}".format(prev_prev_prev_token, prev_prev_token, prev_token, token)
                if tetragram not in tetragram_dict_code:
                    tetragram_dict_code[tetragram] = 1
                else:
                    tetragram_dict_code[tetragram]+=1
                prev_prev_prev_token = prev_prev_token
                prev_prev_token = prev_token
                prev_token = token
            line = f_1.readline()

    sorted_list_code = sorted(tetragram_dict_code.items(), key=lambda item: item[1], reverse=True)
    return sorted_list_code

def remove_unigrams_from_line(input_line, list_of_removable_tokens):
    input_line_split = input_line.split()
    new_input_line = []
    for token in input_line_split:
        if token in list_of_removable_tokens:
            new_input_line += ["<PAD_TOKEN>"]
        else:
            new_input_line += [token]
    return new_input_line

def remove_bigrams_from_line(input_line, list_of_removable_bigrams):
    input_line_split = input_line.split()
    new_input_line = []
    prev_token = ""
    for token in input_line_split:
        bigram = "{}-{}".format(prev_token, token)
        if bigram in list_of_removable_bigrams:
            new_input_line[-1] = "<PAD_TOKEN>"
            new_input_line += ["<PAD_TOKEN>"]
        else:
            new_input_line += [token]
        prev_token = token
    return new_input_line

def remove_trigrams_from_line(input_line, list_of_removable_trigrams):
    input_line_split = input_line.split()
    new_input_line = []
    prev_prev_token = ""
    prev_token = ""
    for token in input_line_split:
        trigram = "{}-{}-{}".format(prev_prev_token, prev_token, token)
        if trigram in list_of_removable_trigrams:
            new_input_line[-2] = "<PAD_TOKEN>"
            new_input_line[-1] = "<PAD_TOKEN>"
            new_input_line += ["<PAD_TOKEN>"]
        else:
            new_input_line += [token]
        prev_prev_token = prev_token
        prev_token = token
    return new_input_line

def remove_tetragrams_from_line(input_line, list_of_removable_tetragrams):
    input_line_split = input_line.split()
    new_input_line = []
    prev_prev_prev_token = ""
    prev_prev_token = ""
    prev_token = ""
    for token in input_line_split:
        tetragram = "{}-{}-{}-{}".format(prev_prev_prev_token, prev_prev_token, prev_token, token)
        if tetragram in list_of_removable_tetragrams:
            new_input_line[-3] = "<PAD_TOKEN>"
            new_input_line[-2] = "<PAD_TOKEN>"
            new_input_line[-1] = "<PAD_TOKEN>"
            new_input_line += ["<PAD_TOKEN>"]
        else:
            new_input_line += [token]
        prev_prev_prev_token = prev_prev_token
        prev_prev_token = prev_token
        prev_token = token
    return new_input_line

def plot_bleu_scores(code_file_path, code_ref_file_path, lang_file_path, lang_ref_file_path):

    # sorted_code_list = get_unigram_list("data/coms-cleaned.train")
    # sorted_lang_list = get_unigram_list("data/natural-language-cleaned-token.en")
    # sorted_code_list = get_bigram_list("data/coms-cleaned.train")
    # sorted_lang_list = get_bigram_list("data/natural-language-cleaned-token.en")
    # sorted_code_list = get_trigram_list("data/coms-cleaned.train")
    # sorted_lang_list = get_trigram_list("data/natural-language-cleaned-token.en")
    sorted_code_list = get_tetragram_list("data/coms-cleaned.train")
    sorted_lang_list = get_tetragram_list("data/natural-language-cleaned-token.en")
    #print(sorted_code_list)
    #print(sorted_lang_list)

    indices = [i for i in range(100)]
    code_bleu_list = []
    lang_bleu_list = []

    for i in range(100):
        in_f1 = open(code_file_path, "r")
        in_f2 = open(code_ref_file_path, "r")
        list_of_removable_tokens = [tuple_link[0] for tuple_link in sorted_code_list[0:i]]
        print(list_of_removable_tokens)
        line1 = in_f1.readline()
        line2 = in_f2.readline()
        total_bleu = 0
        cnt = 0
        while line1:
            line1 = remove_tetragrams_from_line(line1, list_of_removable_tokens)
            sentence_bleu_score = sentence_bleu([line2.split()], line1, weights=(0.25, 0.25, 0.25, 0.25))
            total_bleu = total_bleu + sentence_bleu_score
            line1 = in_f1.readline()
            line2 = in_f2.readline()
            cnt = cnt + 1
        bleu_score = total_bleu / cnt
        code_bleu_list.append(bleu_score)

    for i in range(100):
        in_f1 = open(lang_file_path, "r")
        in_f2 = open(lang_ref_file_path, "r")
        list_of_removable_tokens = [tuple_link[0] for tuple_link in sorted_lang_list[0:i]]
        print(list_of_removable_tokens)
        line1 = in_f1.readline()
        line2 = in_f2.readline()
        total_bleu = 0
        cnt = 0
        while line1:
            line1 = remove_tetragrams_from_line(line1, list_of_removable_tokens)
            sentence_bleu_score = sentence_bleu([line2.split()], line1, weights=(0.25, 0.25, 0.25, 0.25))
            total_bleu = total_bleu + sentence_bleu_score
            line1 = in_f1.readline()
            line2 = in_f2.readline()
            cnt = cnt + 1
        bleu_score = total_bleu / cnt
        lang_bleu_list.append(bleu_score)
    
    plt.title("BLEU-4 scores for source code and natural language with common tetragrams removed")
    plt.xlabel("Number of most tetragrams stripped")
    plt.ylabel("BLEU-4 score")
    plt.plot(indices, code_bleu_list, linewidth=2, color='r')
    plt.plot(indices, lang_bleu_list, linewidth=2, color='b')
    plt.show()

if __name__ == "__main__":
    plot_bleu_scores("reference_translations/coms-translation-small.test", "reference_translations/coms-cleaned-small.test", "reference_translations/decoded_nlp.dec", "reference_translations/newstest2013.en")
    #print(remove_tetragrams_from_line("Here is a sample sentence there.", ["Here-is-a-sample", "is-a-sample-sentence"]))