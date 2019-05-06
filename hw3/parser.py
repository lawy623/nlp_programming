#! /usr/bin/python

__author__="Yue Luo <yl4003@columbia.edu>"
__date__ ="$Mar 31, 2019"

import sys, json, os
from collections import defaultdict
import numpy as np
import math, time


"""
Using Python3.
Q4: Replace the infrequent words at the fringe of the tree.
Q5: Get optimal parsing tree on dev_file based on RARE_Counts.
Q6: Get optimal parsing tree on dev_file based on vert_RARE_Counts.

"""

NEGINF = -9999999999.0

class Counts:
    """
    Class to store the counts of the words, and get the list of infreq words.
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.words = defaultdict(int)

    def count(self, tree):
        """
        Count the frequencies of words and rules in the tree.
        """
        if isinstance(tree, str): return

        if len(tree) == 3:
            self.count(tree[1])
            self.count(tree[2])
        elif len(tree) == 2:
            # It is a unary rule.
            y1 = tree[1] # word
            self.words[y1] += 1

    def replace_rare(self, tree):
        """
        Get to the end of the line. Replace the rare word if needed.
        """
        if len(tree) == 3:
            self.replace_rare(tree[1])
            self.replace_rare(tree[2])
        elif len(tree) == 2:
            y1 = tree[1] # word
            if(self.words[y1]<5):
                tree[1] = "_RARE_"

    def write_rare_file(self, output):
        """
        Write out the file with _RARE_ replacement.
        """
        tag = "_RARE_"
        file_to_replace = open(self.file_name, "r")
        file_to_replace.seek(0)
        file_output = open(output, "w")
        l = file_to_replace.readline()
        count_line = 0
        while l:
            count_line += 1
            t = json.loads(l)
            self.replace_rare(t)
            file_output.write(json.dumps(t)+'\n')
            l = file_to_replace.readline()
        file_output.close()
        file_to_replace.close()
        assert count_line == 5322


class Rule_Counts:
    """
    Class to store the counts of the rules.
    """
    def __init__(self, rare_counts):
        self.count_file = rare_counts
        self.count_X = defaultdict(int)
        self.count_X_YZ = defaultdict(int)
        self.count_X_w = defaultdict(int)
        self.n_nonterminal = 0
        self.nonterminal = []
        self.nonterminal_ind = defaultdict(int)
        self.freq_words = defaultdict(int)
        self.YZ_rules_on_X = dict()

    def get_counts(self):
        """
        Get the counts of the X, X->YZ and X->w terms.
        """
        file = open(self.count_file, 'r')
        l = file.readline()
        while l:
            line = l.strip()
            fields = line.split(" ")
            if fields[1] == "NONTERMINAL":
                self.count_X[fields[2]] = int(fields[0])
            elif fields[1] == "UNARYRULE":
                self.count_X_w[(fields[2],fields[3])] = int(fields[0])
                self.freq_words[fields[3]] = 1
            elif fields[1] == "BINARYRULE":
                self.count_X_YZ[(fields[2],fields[3],fields[4])] = int(fields[0])
                if fields[2] in self.YZ_rules_on_X:
                    self.YZ_rules_on_X[fields[2]].append([fields[3],fields[4]])
                else:
                    self.YZ_rules_on_X[fields[2]] = [[fields[3],fields[4]]]
            l = file.readline()
        self.n_nonterminal = len(self.count_X)
        for k,v in self.count_X.items():
            self.nonterminal.append(k)
        for idx, term in enumerate(self.nonterminal):
            self.nonterminal_ind[term] = idx

    def test_rule_YZ_on_X(self):
        """
        Helper function to test the rules YZ on X.
        """
        count_rule = 0
        for X in self.YZ_rules_on_X:
            print("X:%s (%i)"%(X,len(self.YZ_rules_on_X[X])))
            count_rule += len(self.YZ_rules_on_X[X])
            for i, key in enumerate(self.YZ_rules_on_X[X]):
                print("     i:%i == %s->%s %s"%(i, X, key[0], key[1]))
        print("Total rule:%i"%count_rule)

    def replace_rare(self, w):
        """
        Return the replace word for infreq word.
        """
        return "_RARE_"

    def get_log_q_X_w(self, X, w):
        """
        Get log_q(X->w) value.
        X: non-terminal term
        w: word
        """
        # Replace with infrequent word.
        if self.freq_words[w] == 0:
            w = self.replace_rare(w)

        if self.count_X_w[(X,w)] == 0:
            return NEGINF
        else:
            log_pr = math.log(self.count_X_w[(X,w)]) - math.log(self.count_X[X])
            return log_pr

    def get_log_q_X_YZ(self, X, Y, Z):
        """
        Get the log_q(X->YZ) term.
        XYZ: all are nonterminal terms
        """
        if self.count_X_YZ[(X,Y,Z)] == 0:
            return NEGINF
        else:
            log_pr = math.log(self.count_X_YZ[(X,Y,Z)]) - math.log(self.count_X[X])
            return log_pr

    def build_tree(self, words, dp, bp, start, end, max_X):
        """
        Build the tree recursively.
        """
        # print("[%i-%i]:%s(%i)"%(start,end,max_X,self.nonterminal_ind[max_X]))
        # print(bp[start][end][self.nonterminal_ind[max_X]])
        # print(dp[start,end,self.nonterminal_ind[max_X]])
        tree = []
        if start == end: # leaf node.
            tree.append(max_X)
            tree.append(words[start])
        else:
            tree.append(max_X) # Node's val
            tree.append(self.build_tree(words, dp, bp, start, bp[start][end][self.nonterminal_ind[max_X]][0], bp[start][end][self.nonterminal_ind[max_X]][1])) # Build left tree
            tree.append(self.build_tree(words, dp, bp, bp[start][end][self.nonterminal_ind[max_X]][0]+1, end, bp[start][end][self.nonterminal_ind[max_X]][2])) # Build right tree

        return tree


    def recover_tree(self, words, dp, bp):
        """
        Recover the tree from dp result and bp records.
        Return a tree(list).
        """
        n = len(bp[0])
        n_non = self.n_nonterminal

        ## Get first non-terminal
        max_val = dp[0,n-1,self.nonterminal_ind["S"]]
        max_X = "S"
        if dp[0,n-1,self.nonterminal_ind["S"]] < -9999999.0: # S is not the start symbol. Get one from others if they are good.
            for x in self.nonterminal_ind:
                if max_val < dp[0,n-1,self.nonterminal_ind[x]]:
                    max_val = dp[0,n-1,self.nonterminal_ind[x]]
                    max_X = x

        tree = self.build_tree(words, dp, bp, 0, n-1, max_X)
        return tree

    def CKY(self, line):
        """
        Using CKY algorithm to parse the optimal tree. Use log version.
        Input is a line of sentence.
        Output is a tree(list format).
        """
        words = line.split(" ")
        n = len(words)
        assert n > 0
        n_non = self.n_nonterminal

        dp = np.zeros((n,n,n_non))
        bp = [[[[] for col in range(n_non)] for col in range(n)] for row in range(n)]

        # self.test_rule_YZ_on_X()

        # Forward Pass to get the optimal values.
        ## Initialization
        for i in range(n):
            for idx, term in enumerate(self.nonterminal):
                dp[i,i,idx] = self.get_log_q_X_w(term, words[i])

        ## Iter steps
        for l in range(1,n):
            for i in range(n-l):
                j = i + l
                for X in self.nonterminal_ind:
                    if X in self.YZ_rules_on_X: # X can be further split.
                        ## Set first term X to be compared.
                        Y1Z1 = self.YZ_rules_on_X[X][0]
                        max_log_pr = self.get_log_q_X_YZ(X, Y1Z1[0], Y1Z1[1]) + dp[i,i,self.nonterminal_ind[Y1Z1[0]]] + dp[i+1,j,self.nonterminal_ind[Y1Z1[1]]]
                        sum = 0
                        max_s = i
                        max_Y = Y1Z1[0]
                        max_Z = Y1Z1[1]
                        for YZ in self.YZ_rules_on_X[X]:
                            Y = YZ[0]
                            Z = YZ[1]
                            rule_pr = self.get_log_q_X_YZ(X,Y,Z)
                            for s in range(i, j):
                                sum =  rule_pr + dp[i,s,self.nonterminal_ind[Y]] + dp[s+1,j,self.nonterminal_ind[Z]]
                                if sum > max_log_pr:
                                    max_log_pr = sum
                                    max_s = s
                                    max_Y = Y
                                    max_Z = Z
                        dp[i,j,self.nonterminal_ind[X]] = max_log_pr
                        bp[i][j][self.nonterminal_ind[X]] = [max_s,max_Y,max_Z]
                    else: # X can not be split any more.
                        dp[i,j,self.nonterminal_ind[X]] = NEGINF
                        bp[i][j][self.nonterminal_ind[X]] = [-1," "," "]

        return self.recover_tree(words, dp, bp)

    def write_output(self, dev_file, pred_file):
        """
        Using the counter, write predicted parsing tree for the dev file.
        """
        dev_reader = open(dev_file, 'r')
        pred_writer = open(pred_file, 'w')
        dev_reader.seek(0)

        l = dev_reader.readline()
        while l:
            line = l.strip()
            tree = self.CKY(line)
            pred_writer.write(json.dumps(tree)+"\n")
            l = dev_reader.readline()
        dev_reader.close()
        pred_writer.close()

def usage():
    print("""
    python parser.py [prob] [...]
        [Prob]:
            q4: [q4] [input_parse] [output]
                Read in a training input file, and replace all low frequency word by _RARE_.
            q5: [q5] [input_parse_rare] [dev_file] [pred_file]
                Read in a training(_RARE_), a dev_file, and produce the pred result.
            q6: [q6] [input_parse_vert_rare] [dev_file] [pred_file]
                Read in a training(vert_RARE_), a dev_file, and produce the pred result.
    """)

def main(prob):
    if prob == "q4":
        parse_file = sys.argv[2]
        output_file = sys.argv[3]

        counter_word = Counts(parse_file)
        for l in open(parse_file):
            t = json.loads(l)
            counter_word.count(t)

        counter_word.write_rare_file(output_file)
    elif prob == "q5":
        start_time = time.time()
        parse_rare_file = sys.argv[2]
        dev_file = sys.argv[3]
        pred_file = sys.argv[4]
        if(not os.path.exists(parse_rare_file)):
            sys.stderr.write("ERROR: RARE Training not found.\n" )
            sys.exit(1)
        if(not os.path.exists(dev_file)):
            sys.stderr.write("ERROR: Dev file not found.\n" )
            sys.exit(1)

        print("Running count file on _RARE_ Training...")
        try:
            os.system('python3 count_cfg_freq3.py parse_train.RARE.dat > cfg.RARE.counts')
        except OSError:
            sys.stderr.write("ERROR: Cannot create countfile.\n" )
            sys.exit(1)

        rare_counts = "cfg.RARE.counts"
        counter_rule = Rule_Counts(rare_counts)
        counter_rule.get_counts()
        counter_rule.write_output(dev_file, pred_file)
        print("Total time: %.4fs..." % ((time.time()-start_time)))
    elif prob == "q6":
        start_time = time.time()
        parse_rare_file = sys.argv[2]
        dev_file = sys.argv[3]
        pred_file = sys.argv[4]
        if(not os.path.exists(parse_rare_file)):
            sys.stderr.write("ERROR: RARE Training not found.\n" )
            sys.exit(1)
        if(not os.path.exists(dev_file)):
            sys.stderr.write("ERROR: Dev file not found.\n" )
            sys.exit(1)

        print("Running count file on vert_RARE_ Training...")
        try:
            os.system('python3 count_cfg_freq3.py parse_train_vert.RARE.dat > cfg_vert.RARE.counts')
        except OSError:
            sys.stderr.write("ERROR: Cannot create countfile.\n" )
            sys.exit(1)

        rare_counts = "cfg_vert.RARE.counts"
        counter_rule = Rule_Counts(rare_counts)
        counter_rule.get_counts()
        counter_rule.write_output(dev_file, pred_file)
        print("Total time: %.4fs..." % ((time.time()-start_time)))
    else:
        usage()

if __name__ == "__main__":
    main(sys.argv[1])
