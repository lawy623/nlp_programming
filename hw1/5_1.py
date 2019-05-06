#! /usr/bin/python

__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$Feb 16, 2019"

import sys
from collections import defaultdict
import math

"""
Calculate the q(w|u,v) terms based on the counting. Should include STOP, * also.
Input should be a trigram.txt where each line is w_{i-2}, w_{i-1}, w_{i}.
"""

def get_counts(count_file):
    """
    Get counts for trigrams and bigrams.
    """
    Count_trigram = defaultdict(int) # Use Dict to store all counts.
    Count_bigram = defaultdict(int)
    l = count_file.readline()
    while l:
        line = l.strip()
        fields = line.split(" ")
        if fields[1] == "2-GRAM":
            Count_bigram[(fields[2],fields[3])] = int(fields[0])
        elif fields[1] =="3-GRAM":
            Count_trigram[(fields[2],fields[3],fields[4])] = int(fields[0])
        l = count_file.readline()
    return Count_trigram, Count_bigram

def cal_trigram_param(Count_trigram, Count_bigram, fields):
    """
    Calculate the log likelihood of q(w|n,v). Fields should be in the order
    of w_{i-2}, w_{i-1}, w_{i}.
    If trigram or bigram is not found in the training set, log prob will be set
    to -infty.
    """
    if (fields[0],fields[1],fields[2]) in Count_trigram:
        count_3 = Count_trigram[(fields[0],fields[1],fields[2])]
    else: # Trigram not found in Training set
        print "Trigram [%s,%s,%s] not found" % (fields[0],fields[1],fields[2])
        return -9999999999.0

    if (fields[0],fields[1]) in Count_bigram:
        count_2 = Count_bigram[(fields[0],fields[1])]
    else: # Bigram not found in Training set
        print "Bigram [%s,%s] not found" % (fields[0],fields[1])
        return -9999999999.0

    log_pr = math.log(count_3) - math.log(count_2)
    assert log_pr <= 0
    return log_pr

def write_output(Count_trigram, Count_bigram, input_file, output_name):
    """
    Based on the counts and the input file, calculate log likelihood.
    Write the trigram, log likelihood to output_file.
    """
    output_file = file(output_name, "w")
    input_file.seek(0)
    l = input_file.readline()
    while l:
        line = l.strip()
        fields = line.split(" ")
        assert len(fields)==3
        log_pr = cal_trigram_param(Count_trigram, Count_bigram, fields) # Calculate using naive estimator.
        l = line + " " + str(log_pr) + "\n"
        output_file.write(l)
        l = input_file.readline()
    output_file.close()

def usage():
    print """
    python 5_1.py
        Read a trigram from lines in a file, and get the log likelihood of it.
        No arguments should be taken.
    """

if __name__ == "__main__":

    if len(sys.argv)>=2:
        usage()
        sys.exit(2)

    count_name = "./ner_rare.counts" # "./ner_rare.counts" is the same
    input_name = "./trigrams.txt"

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

    # Get the bigram and trigram counts.
    Count_trigram, Count_bigram = get_counts(count_file)

    # Get trigram's log likelihood and output it.
    output_name = "5_1.txt"
    write_output(Count_trigram, Count_bigram, input_file, output_name)
