#! /usr/bin/python

__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$Feb 16, 2019"

import sys
from collections import defaultdict
import math
from count_freqs import sentence_iterator,simple_conll_corpus_iterator
import numpy as np
import time

"""
Implement the Modified Viterbi Algorithm and runs it on the ner_dev.data.
We import the iterator method from the "count_freqs.py" file.
"""

class counter(object):
    """
    The counter stores all the counts for word, tags, trigram and bigram.
    """
    def __init__(self, count_file, all_tags):
        self.count_file = count_file
        self.Count_y_x = defaultdict(int)
        self.Count_y = defaultdict(int)
        self.Count_trigram = defaultdict(int)
        self.Count_bigram = defaultdict(int)
        self.all_tags = all_tags
        self.trigram = defaultdict(int)
        self.frequency_word = defaultdict(int)

    def get_counts(self):
        """
        Get the counts that will be used to compute the q and e's used in Viterbi
        algorithm.
        q = Count_trigram/Count_bigram
        e = Count_y_x / Count_y
        """
        l = count_file.readline()
        while l:
            line = l.strip()
            fields = line.split(" ")
            if fields[1] == "WORDTAG":
                self.Count_y_x[(fields[2],fields[3])] = int(fields[0])
                self.frequency_word[fields[3]] += int(fields[0])
            elif fields[1] =="1-GRAM":
                self.Count_y[fields[2]] = int(fields[0])
            elif fields[1] == "2-GRAM":
                self.Count_bigram[(fields[2],fields[3])] = int(fields[0])
            elif fields[1] =="3-GRAM":
                self.Count_trigram[(fields[2],fields[3],fields[4])] = int(fields[0])
            l = count_file.readline()

    def get_log_e_x_v(self, word, tag):
        """
        Get the log_e(x|v) value give x=word and v=tag.
        Value is Count(v->x) / Count(v).
        First consider whether this word is _RARE_.
        """
        # First need to consider whether this word is _RARE_.
        assert tag in self.all_tags # Ensure that this is a value tag.
        if word not in self.frequency_word:
            word = "_RARE_"

        if counter.Count_y_x[(tag,word)] == 0: # This word, tag combination has never appeared.
            # print "Tag %s not exist for this word..." % tag
            return -9999999999.0

        log_pr = math.log(counter.Count_y_x[(tag,word)]) - math.log(counter.Count_y[tag]) # same as log(a/b)
        assert log_pr <= 0
        return log_pr

    def get_log_q_v_w_u(self, tags):
        """
        Get the log_q(v|w,u) value give w = w_{i-2}, u=w_{i-1} and v=w_{i}
        Tags are in order w, u, v.
        Value is Count(w,u,v) / Count(w,u).
        """
        if (tags[0],tags[1],tags[2]) in counter.Count_trigram:
            count_3 = counter.Count_trigram[(tags[0],tags[1],tags[2])]
        else: # Trigram not found in Training set
            # print "Trigram [%s,%s,%s] not found" % (tags[0],tags[1],tags[2])
            return -9999999999.0

        if (tags[0],tags[1]) in counter.Count_bigram:
            count_2 = counter.Count_bigram[(tags[0],tags[1])]
        else: # Bigram not found in Training set
            # print "Bigram [%s,%s] not found" % (tags[0],tags[1])
            return -9999999999.0

        log_pr = math.log(count_3) - math.log(count_2)
        assert log_pr <= 0
        return log_pr

    def get_trigrams(self, trigram_name):
        """
        Directly get the trigram from 5_1.txt. But not use in this work.
        """
        try:
            trigram_file = file(trigram_name,"r")
        except IOError:
            sys.stderr.write("ERROR: Cannot read trigramfile %s.\n" % trigram_name)
            sys.exit(1)

        trigram_file.seek(0)
        l = trigram_file.readline()
        while l:
            line = l.strip()
            fields = line.split(" ")
            self.trigram[(fields[0],fields[1],fields[2])] = float(fields[3])
            l = trigram_file.readline()

    def get_log_q_v_w_u_byfile(self, tags):
        """
        Getting the q terms from the pre-calculated results.  But not use in this work.
        """
        if (tags[0],tags[1],tags[2]) in self.trigram:
            return self.trigram[(tags[0],tags[1],tags[2])]
        else:
            return -9999999999.0

def viterbi(sentence, counter):
    """
    Calculate the tages of a sentence based on the counts from training data.
    Using modified Viterbi Algorithm with log probability.
    Output a list of tags with the same size as sentence, and the log prob.
    """
    n = len(sentence)
    assert n > 0 # Should be a valid sentence.
    n_tags = len(counter.all_tags)
    tags = []
    log_pr = []

    dp = np.zeros((n,n_tags,n_tags))
    bp = np.zeros((n,n_tags,n_tags))

    # Forward Process to get the table
    for i in range(n):
        if i == 0: # This case only with (*,*,_) trigram.
            for j in range(n_tags): # iter all v. w and u are fixed as *. No need for bp[].
                v = counter.all_tags[j]
                w_u_v = ["*","*",v]
                sum = counter.get_log_q_v_w_u(w_u_v) + counter.get_log_e_x_v(sentence[i],v)
                dp[0,0,j] = sum
        elif i == 1: # This case only with (*,_,_) trigram.
            for j in range(n_tags):
                for k in range(n_tags):
                    u = counter.all_tags[j]
                    v = counter.all_tags[k]
                    w_u_v = ["*",u,v] # w is always fixed as *. No need for bp[].
                    sum = dp[0,0,j] + counter.get_log_q_v_w_u(w_u_v) + counter.get_log_e_x_v(sentence[i],v)
                    dp[i,j,k] = sum
        else:
            for j in range(n_tags):
                for k in range(n_tags):
                    u = counter.all_tags[j]
                    v = counter.all_tags[k]
                    w = counter.all_tags[0]
                    w_u_v = [w,u,v]
                    sum = dp[i-1,0,j] + counter.get_log_q_v_w_u(w_u_v) + counter.get_log_e_x_v(sentence[i],v)
                    max_log_pr = sum # Set to the first one. Otherwise is a bug here.
                    max_tag = 0
                    emission = counter.get_log_e_x_v(sentence[i],v) # Reduce time.
                    for l in range(n_tags):
                        w = counter.all_tags[l]
                        w_u_v = [w,u,v] # this should be a triple set now.
                        sum = dp[i-1,l,j] + counter.get_log_q_v_w_u(w_u_v) + emission
                        if sum > max_log_pr:
                            max_log_pr = sum
                            max_tag = l
                    dp[i,j,k] = max_log_pr
                    bp[i,j,k] = max_tag

    # Backward process to retrieve the tags.
    if n == 1: # Only one word in sequence.
        max_tag = 0
        max_log_pr = -9999999999.0
        for j in range(n_tags):
            v = counter.all_tags[j]
            w_u_v = ["*",v,"STOP"]
            sum = dp[0,0,j] + counter.get_log_q_v_w_u(w_u_v)
            if sum > max_log_pr:
                max_log_pr = sum
                max_tag = j
        tags = [counter.all_tags[j]]
        log_pr = [max_log_pr]
    else: # Get the last two tags first.
        max_tag_u = 0
        max_tag_v = 0
        max_log_pr = -9999999999.0
        for j in range(n_tags):
            for k in range(n_tags):
                u = counter.all_tags[j]
                v = counter.all_tags[k]
                w_u_v = [u,v,"STOP"]
                sum = dp[n-1,j,k] + counter.get_log_q_v_w_u(w_u_v)
                if sum > max_log_pr:
                    max_log_pr = sum
                    max_tag_u = j
                    max_tag_v = k
        tags.append(counter.all_tags[max_tag_v])
        tags.append(counter.all_tags[max_tag_u])
        log_pr.append(dp[n-1,max_tag_u,max_tag_v])

        # Iterate back and find previous tags.
        for i in range(n-3,-1,-1):
            tags.append(counter.all_tags[int(bp[i+2,max_tag_u,max_tag_v])])
            max_tag_v = max_tag_u
            max_tag_u = int(bp[i+2,max_tag_u,max_tag_v])
            log_pr.append(dp[i+1,max_tag_u,max_tag_v])
        log_pr.append(dp[0,0,max_tag_u])

    # Need to reverse the two lists since we store in reversed order.
    tags.reverse()
    log_pr.reverse()

    assert len(tags) == n and len(log_pr) == n
    return tags, log_pr

def write_output(counter, input_file, output_name):
    """
    Based on the counts and the input file, calculate the most likely tag using
    viterbi algorithm, and write the word, the predicted tag and log likelihood
    to output_file.
    Dev.dat should have 3247 sentences.
    """
    output_file = file(output_name, "w")
    input_file.seek(0)
    start_time = time.time()

    sentence_iter = sentence_iterator(simple_conll_corpus_iterator(input_file))
    count = 0
    for sentence in sentence_iter: # run viterbi for each sentence.
        count += 1
        s = [x[1] for x in sentence]
        tags, log_pr = viterbi(s, counter)
        for i in range(len(s)):
            l = s[i] + " " + tags[i] + " " + str(log_pr[i]) + "\n"
            output_file.write(l)
        output_file.write("\n")
    output_file.close()

    print "Total time: %.4fs...%i sentences" % ((time.time()-start_time),count)


def usage():
    print """
    python 5_2.py
        Read a count_file which already counted all low freq word as _RARE_. And
        a input_file with words you want to label with HMM model and Viterbi Algo.
        No arguments should be taken.
    """

if __name__ == "__main__":

    if len(sys.argv)>=2:
        usage()
        sys.exit(2)

    count_name = "./ner_rare.counts"
    input_name = "./ner_dev.dat"

    # trigram_name = "./5_1.txt" # Not use it in this work.

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

    all_tags = ["I-PER","I-ORG","B-ORG","I-LOC","B-LOC","I-MISC","B-MISC","O"]
    # Get the counts
    counter = counter(count_file, all_tags)
    counter.get_counts()

    # Run estimator on Dev file and output it.
    output_name = "5_2.txt"
    write_output(counter, input_file, output_name)
    # After that, should use "python eval_ne_tagger.py ner_dev.key 5_2.txt" to check.
