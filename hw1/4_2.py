#! /usr/bin/python

__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$Feb 16, 2019"

import sys
from collections import defaultdict
import math

"""
Before running this, you should already have the new count table with the
_RARE_ tags. Then input both the count file and the data to be labelled.

This will run the naive estimator for tagging.
"""

def get_counts(count_file):
    """
    Get the counts for the NE, and 1-gram, which will be used to
    get the emission parameters. Return Count(y->x) and Count(y).
    """
    Count_y_x = defaultdict(int) # Use Dict to store all counts.
    Count_y = defaultdict(int)
    l = count_file.readline()
    while l:
        line = l.strip()
        fields = line.split(" ")
        if fields[1] == "WORDTAG":
            Count_y_x[(fields[2],fields[3])] = int(fields[0])
        elif fields[1] =="1-GRAM":
            Count_y[fields[2]] = int(fields[0])
        l = count_file.readline()
    return Count_y_x, Count_y

def naive_estimator(Count_y_x, Count_y, word):
    """
    Calculate the most likely tag by naively picking the greatest probability.
    Number of possible tags based on 1-Gram is 8.
    """
    tags = ["I-PER","I-ORG","B-ORG","I-LOC","B-LOC","I-MISC","B-MISC","O"]
    zero_count = 0
    max_tag = ""
    max_log_pr = -9999999999.0
    for i in tags:
        if Count_y_x[(i,word)] == 0:
            zero_count += 1
        else:
            if (math.log(Count_y_x[(i,word)]) - math.log(Count_y[i]) > max_log_pr):
                max_log_pr = math.log(Count_y_x[(i,word)]) -math.log(Count_y[i])
                max_tag = i

    if zero_count == len(tags): # Meaning that this word is _RARE_ class. Since can not found it with any tag.
        word = "_RARE_"

        max_tag = ""
        max_log_pr = -9999999999.0
        for i in tags:
            if Count_y_x[(i,word)] == 0:
                pass
            else:
                if (math.log(Count_y_x[(i,word)]) - math.log(Count_y[i]) > max_log_pr):
                    max_log_pr = math.log(Count_y_x[(i,word)]) -math.log(Count_y[i])
                    max_tag = i

    assert max_tag != "" and max_log_pr != -9999999999.0 # Assert a class should be assigned.
    return max_tag, max_log_pr

def write_output(Count_y_x, Count_y, input_file, output_name):
    """
    Based on the counts and the input file, calculate the most likely tag using
    emission values, and write the word, the predicted tag and log likelihood
    to output_file.
    """
    output_file = file(output_name, "w")
    input_file.seek(0)
    l = input_file.readline()
    while l:
        if l.strip(): #for non empty line
            word = l.strip()
            tag, log_pr = naive_estimator(Count_y_x, Count_y, word) # Calculate using naive estimator.
            l = " ".join([word, tag, str(log_pr),"\n"])
        output_file.write(l)
        l = input_file.readline()
    output_file.close()


def usage():
    print """
    python 4_2.py
        Read a count_file which already counted all low freq word as _RARE_. And
        a input_file with words you want to label with this naive estimator.
        No arguments should be taken.
    """

if __name__ == "__main__":

    if len(sys.argv)>=2:
        usage()
        sys.exit(2)

    count_name = "./ner_rare.counts"
    input_name = "./ner_dev.dat"

    try:
        count_file = file(count_name,"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read countfile %s.\n" % count_name)
        sys.exit(1)

    try:
        input_file = file(input_name,"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % input_name)
        sys.exit(1)

    # Get the counts
    Count_y_x, Count_y = get_counts(count_file)

    # Run estimator on Dev file and output it.
    output_name = "4_2.txt"
    write_output(Count_y_x, Count_y, input_file, output_name)
    # After that, should use "python eval_ne_tagger.py ner_dev.key 4_2.txt" to check.
