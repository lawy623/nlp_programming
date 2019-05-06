#! /usr/bin/python

__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$Feb 16, 2019"

import sys
from collections import defaultdict

"""
We should assume that we do not have the ner.counts file at this moment.

So we design the algorithm directly based on the original ner_train.dat.
"""

class rare_file(object):
    """
    Store original file, counts for words.
    """
    def __init__(self, input):
        self.file = input
        self.counts = defaultdict(int)

    def get_count(self):
        """
        Get a dict for counting the frequency of all the word in the file.
        Input is the open read file, output a dict.
        """
        self.file.seek(0) # Move back to first place
        l = self.file.readline()
        while l:
            line = l.strip() # Remove the "\n" at the end of line.
            if line: #for non empty line
                word = " ".join(line.split(" ")[:-1])
                self.counts[word] += 1
            l = self.file.readline()

    def write_rare_file(self, output):
        """
        Write out the file with _RARE_ replacement.
        """
        tag = "_RARE_"
        self.file.seek(0) # Move back to first place
        file_output = file(output, "w")
        l = self.file.readline()
        while l:
            if l.strip(): #for non empty line
                word = " ".join(l.split(" ")[:-1])
                if self.counts[word] < 5: #replace <5 WORDS as _RARE_
                    l = " ".join([tag, l.split(" ")[-1]])
            file_output.write(l)
            l = self.file.readline()
        file_output.close()

    def test_counter(self, count1, count2):
        """
        self test for counter's content.
        Should have count1 for >=5 word. count2 for <5 word.
        """
        n1 = 0
        n2 = 0
        for i in self.counts:
            if self.counts[i] >= 5:
                n1 += 1
            else:
                n2 += 1
        assert n1 == count1 and n2 == count2


def usage():
    print """
    python 4_1.py
        Read in a name entity tagged training input file, and replace all
        low frequency word by _RARE_. No arguments should be taken.
    """

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        usage()
        sys.exit(2)

    input_name = "./ner_train.dat"
    try:
        input = file(input_name,"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % input_name)
        sys.exit(1)


    # init and get the counts
    rare = rare_file(input)
    rare.get_count()

    # print out file with _RARE_ Replacement
    output_name = "./ner_train_rare.dat"
    rare.write_rare_file(output_name)
