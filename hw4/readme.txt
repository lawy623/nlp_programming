This programming homework is using python 2.7.15, tensorflow 1.12.0, numpy 1.15.2. Develop and run on my Mac Pro.

Before training, run the following under dir 'nlp_hw_dep' and run the following command to generate the vocabulary and data(as described in the guidance):
'python src/gen vocab.py trees/train.conll data/vocabs'
'python src/gen.py trees/train.conll data/train.data'
'python src/gen.py trees/dev.conll data/dev.data'
There will create 6 files in /data folder.

All the changes for this assignment are in the file 'src/train.py' and 'src/depModel.py'.


Part1
All parameters are hardcoded in 'src/train.py'.
To run the training, ensure 'h1_dim' and 'h2_dim' are set as 200 are required in the question. And model_dir is set as './models/q1'.
Then run 'python2 src/train'.
It will ends after 7 epochs. We saved the checkpoints at the end of each epoch. Training is roughly 15min-20min on my macbook pro. Model is saved in
'/models/q1/check_points'.
For testing, I only upload the best model among them. It is saved after the 6 epoch(model.ckpt-5). At the 7 epoch it starts to overfit on the training data.
Run 'python2 src/depModel.py trees/dev.conll outputs/dev_part1.conll q1 5' to run the NN parser on dev data.
Then run 'python2 src/eval.py trees/dev.conll outputs/dev_part1.conll' to see the result on dev data.
---------------------------------
Unlabeled attachment score 82.62
Labeled attachment score 78.66
---------------------------------
We can see that for both UAS and LAS we meet the requirement.
Run 'python2 src/depModel.py trees/test.conll outputs/test_part1.conll q1 5', we get the result on test data.
All these outputs are in the '/outputs' folder.



Part2
Similar as in Part1, but change 'h1_dim' and 'h2_dim' to be 400. And set model_dir as './models/q2'.
Then run 'python2 src/train'.
It will ends after 7 epochs. We saved the checkpoints at the end of each epoch. Training is roughly 15min-20min on my macbook pro. Model is saved in
'/models/q2/check_points'.
For testing, I only upload the best model among them. It is saved after the 6 epoch(model.ckpt-5). At the 7 epoch it starts to overfit on the training data.
Run 'python2 src/depModel.py trees/dev.conll outputs/dev_part2.conll q2 5' to run the NN parser on dev data.
Then run 'python2 src/eval.py trees/dev.conll outputs/dev_part2.conll' to see the result on dev data.
---------------------------------
Unlabeled attachment score 83.44
Labeled attachment score 79.61
---------------------------------
We can see that for both UAS and LAS we meet the requirement. It has been an improvement of nearly one additional score for both UAS and LAS. These
changes are directly from the increase of the model size(i.e. set the last two hidden layers with more units). This larger model contains more parameters,
which help to better model the features into the final action. The model capacity has been improved so that the scores are higher now.


