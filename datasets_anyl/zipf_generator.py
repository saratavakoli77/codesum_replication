import math
import re
import matplotlib.pyplot as plt
import pickle
import argparse
from decal_custom.eval_funcs import get_tokens_for_dataset_examples
from decal_custom.data_hardcode import relevant_datasets, get_dataset_examples

delimiters = ";", ",", " ", ".", "\n"
regexPattern = '|'.join(map(re.escape, delimiters))

"""
This file creates the plots seen in figures (1) and (2) in the associated paper.
Example usage: python zipf_generator.py --ngram unigram
                                        --data_file_1 zipf_plot_data/io/funcom.txt
                                        --data_file_2 zipf_plot_data/io/nlp.txt

It essentially creates zipf plots: in this context, the plot the log frequency of
n grams in different datasets. The data files used are in the folder `zipf_plot_data`,
and are .txt files.
NOTE: all this data has already been collected for you in .p (pickle) files, also in
`zipf_plot_data/pickle`.
"""

def plot_zipf(title, x_label, y_label, x_axis_code, y_axis_code, x_axis_lang, y_axis_lang):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_axis_code, y_axis_code, linewidth=2, color='r')
    plt.plot(x_axis_lang, y_axis_lang, linewidth=2, color='b')
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

def process_and_plot(processing_func, n_gram, input_file_1, input_file_2):
    dict_input, input_token_count = processing_func(input_file_1, {})
    dict_output, output_token_count = processing_func(input_file_2, {})
    
    """
    Here, we sort the ngrams by frequency, from largest to smallest.
    """
    sorted_list_input = sorted(dict_input.items(), key=lambda item: item[1], reverse=True)
    sorted_list_output = sorted(dict_output.items(), key=lambda item: item[1], reverse=True)

    """
    Create indices for the plot, and divide by token count (total number of ngrams in dataset)
    to get a true probability distribution.
    """
    x_axis_input = [x+1 for x in range(len(sorted_list_input))]
    y_axis_input = [[1.0 * item[1]/input_token_count] for item in sorted_list_input]

    x_axis_output = [x+1 for x in range(len(sorted_list_output))]
    y_axis_output = [[1.0 * item[1]/output_token_count] for item in sorted_list_output]

    plot_zipf(n_gram, "Log Freq Rank", "Log Freq", x_axis_input, y_axis_input, x_axis_output, y_axis_output)

def plot_unigram_zipf(input_file_1, input_file_2):
    #def process_unigrams(input_file, dict):
    #    with open(input_file) as f:
    #        line = f.readline()
    #        token_count = 0
    #        while line:
    #            for token in get_tokens_for_dataset_examples(line, ):
    #                token_count+=1
    #                if token == "\n": # stop once you've reached a new line
    #                    break
    #                if token == "": # don't include NULL tokens
    #                    continue
    #                if token not in dict:
    #                    dict[token] = 1
    #                else:
    #                    dict[token]+=1
    #            line = f.readline()
    #    return dict, token_count
    #
    #process_and_plot(process_unigrams, "Unigrams", input_file_1, input_file_2)

def plot_bigram_zipf(input_file_1, input_file_2):

    def process_bigrams(input_file, dict):
        with open(input_file) as f:
            line = f.readline()
            bigram_count = 0
            while line:
                line = re.split(regexPattern, line)
                prev_token = ""
                for token in line:
                    if token == "\n":
                        break
                    if prev_token == "" or token == "":
                        prev_token = token
                        continue
                    bigram_count+=1
                    bigram = "{}-{}".format(prev_token, token)
                    if bigram not in dict:
                        dict[bigram] = 1
                    else:
                        dict[bigram]+=1
                    prev_token = token
                line = f.readline()
        return dict, bigram_count

    process_and_plot(process_bigrams, "Bigrams", input_file_1, input_file_2)

def plot_trigram_zipf(input_file_1, input_file_2):

    def process_trigrams(input_file, dict):
        with open(input_file) as f:
            line = f.readline()
            trigram_count = 0
            while line:
                line = re.split(regexPattern, line)
                prev_token = ""
                prev_prev_token = ""
                for token in line:
                    if token == "\n":
                        break
                    if prev_prev_token == "" or prev_token == "" or token == "":
                        prev_prev_token = prev_token
                        prev_token = token
                        continue
                    trigram_count+=1
                    trigram = "{}-{}-{}".format(prev_prev_token, prev_token, token)
                    if trigram not in dict:
                        dict[trigram] = 1
                    else:
                        dict[trigram]+=1
                    prev_prev_token = prev_token
                    prev_token = token
                line = f.readline() 
        return dict, trigram_count

    process_and_plot(process_trigrams, "Trigrams", input_file_1, input_file_2)

def plot_tetragram_zipf(input_file_1, input_file_2):

    def process_tetragrams(input_file, dict):
        with open(input_file) as f:
            line = f.readline()
            tetragram_count = 0
            while line:
                line = re.split(regexPattern, line)
                prev_token = ""
                prev_prev_token = ""
                prev_prev_prev_token = ""
                for token in line:
                    if token == "\n":
                        break
                    if prev_prev_prev_token == "" or prev_prev_token == "" or prev_token == "" or token == "":
                        prev_prev_prev_token = prev_prev_token
                        prev_prev_token = prev_token
                        prev_token = token
                        continue
                    tetragram_count+=1
                    tetragram = "{}-{}-{}-{}".format(prev_prev_prev_token, prev_prev_token, prev_token, token)
                    if tetragram not in dict:
                        dict[tetragram] = 1
                    else:
                        dict[tetragram]+=1
                    prev_prev_prev_token = prev_prev_token
                    prev_prev_token = prev_token
                    prev_token = token
                line = f.readline()
        return dict, tetragram_count

    process_and_plot(process_tetragrams, "Tetragrams", input_file_1, input_file_2)

def main(args):

    if args.ngram == "unigram":
        plotting_function = plot_unigram_zipf
    elif args.ngram == "bigram":
        plotting_function = plot_bigram_zipf
    elif args.ngram == "trigram":
        plotting_function = plot_trigram_zipf
    else:
        plotting_function = plot_tetragram_zipf

    plotting_function(args.data_file_1, args.data_file_2)

if __name__ == "__main__":
    for dataset in relevant_datasets:
        data = get_dataset_examples(dataset, "train")

    #parser = argparse.ArgumentParser()

    #parser.add_argument("--ngram", choices=["unigram", "bigram", "trigram", "tetragram"], required=True)
    #parser.add_argument("--data_file_1", required=True)
    #parser.add_argument("--data_file_2", required=True)

    #args = parser.parse_args()
    #main(args)







