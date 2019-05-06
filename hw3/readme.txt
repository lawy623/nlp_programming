Q4
Run:
• python3 parser.py q4 parse_train.dat parse_train.RARE.dat



Q5
Run:
• python3 parser.py q5 parse_train.RARE.dat parse_dev.dat q5_prediction_file
• python3 eval_parser3.py parse_dev.key q5_prediction_file > q5_eval.txt

Total time of running: 47.3534s...(On my Macbook)

For the CKY Algorithm, in order to achieve efficient performance, instead of store all the
rules X->YZ, I also use a map to store all possible YZ's of a certain X(as a key). Instead of
finding all YZ(which can be |N|^2), I only find it for all possible YZ that X can be. This
reduces the time.

Over all, the algorithm gives the following results on the dev_file:
Type       Total       Precision     Recall    F1 Score
===============================================================
total      4664            0.713      0.713      0.713
F1 score is 0.713, with the same precision and recall. It is a relative ok result for the parsing.
But checking q5_eval.txt, I found a interesting phenomenon, that is when the total number of NONTERMINAL
is small, the f1 score is relatively lower than the ones that are with a large samples.
(e.g
Type       Total       Precision     Recall    F1 Score
===============================================================
.          370               1.0        1.0        1.0
ADJ        164             0.827      0.555      0.664
ADJP       29              0.333      0.241       0.28
But some NONTERMINAL with small amount is quite good.)
I may conclude this as the problem in the training samples, that the less amount of appearances
may not reflect the whole distribution of the case.



Q6
Run:
• python3 parser.py q4 parse_train_vert.dat parse_train_vert.RARE.dat
• python3 parser.py q6 parse_train_vert.RARE.dat parse_dev.dat q6_prediction_file
• python3 eval_parser3.py parse_dev.key q6_prediction_file > q6_eval.txt

Total time of running: 82.2710s...(On my Macbook)

I don't modify the algorithm in Q5 and just directly apply it here. Since I store all possible YZ
of a single X, the running time is still manageable and within the required limit.
Over all, the algorithm using the new training tags gives the following results on the dev_file:
Type       Total       Precision     Recall    F1 Score
===============================================================
total      4664            0.742      0.742      0.742
F1 score is 0.742, with the same precision and recall. It is better than the result in Q5, mainly
because the 'vertical markovization' helps to capture the relationship between different levels
of the tree, and it provides contextual information for the parsing.
It also slows down the algorithm since the NONTERMINAL's set is larger than before, since
we may therefore need to explore extra possibilities.

Comparing q5_eval and q6_eval, we can see that the f1 score of each individual term most likely
has been improved, and it is especially true for the terms that are with less amount/appearances.

And we compare a simple sample, and check the tree for two different cases:
'The complicated language in the huge new law has muddied the fight .'

Using Q5 and Q6, the scores are
Type       Total       Precision     Recall    F1 Score
===============================================================
total      25               0.92       0.92       0.92          -------- Q5
total      25               0.96       0.96       0.96          -------- Q6
Then 'vertical markovization' helps to improve this single example. The boost comes from
Type       Total       Precision     Recall    F1 Score
===============================================================
NP         7               0.857      0.857      0.857          -------- Q5
NP         7                 1.0        1.0        1.0          -------- Q6
Visualizing the parsing tree of its key/Q5_result/Q6_result, we get:
##### Key:
[S,
 [NP,
  [NP, [DET, The], [NP, [VERB, complicated], [NOUN, language]]],
  [PP,
   [ADP, in],
   [NP, [DET, the], [NP, [ADJ, huge], [NP, [ADJ, new], [NOUN, law]]]]]],
 [S,
  [VP, [VERB, has], [VP, [VERB, muddied], [NP, [DET, the], [NOUN, fight]]]],
  [., .]]]
##### Q5_result:
[S,
 [NP,
  [DET, The],
  [NP,
   [NP, [NOUN, complicated], [NOUN, language]],
   [PP,
    [ADP, in],
    [NP, [DET, the], [NP, [ADJ, huge], [NP, [ADJ, new], [NOUN, law]]]]]]],
 [S,
  [VP, [VERB, has], [VP, [VERB, muddied], [NP, [DET, the], [NOUN, fight]]]],
  [., .]]]
##### Q6_result:
[S,
 [NP^<S>,
  [NP^<NP>, [DET, The], [NP, [NOUN, complicated], [NOUN, language]]],
  [PP^<NP>,
   [ADP, in],
   [NP^<PP>, [DET, the], [NP, [ADJ, huge], [NP, [ADJ, new], [NOUN, law]]]]]],
 [S,
  [VP^<S>,
   [VERB, has],
   [VP^<VP>, [VERB, muddied], [NP^<VP>, [DET, the], [NOUN, fight]]]],
  [., .]]]
For both methods, they perform quite good on this example. But we can see that Q6's
method is better.
For example, using Q6, NP^<S> -> NP^<NP> PP^<NP>,
but using Q5, it is NP -> DET NP, and NP -> NP PP.
Easy to see that the parent's tag helps the parsing in such a case.
