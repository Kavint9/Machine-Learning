The zip file contains the following documents
1) Mult Naive Bayes - Bag of words representaion
	Usage - python <filename> <path_train_spam> <path_train_ham> <path_test_spam> <path_test_ham>

2) Discrete Naive Bayes - Bern representaion
	Usage - python <filename> <path_train_spam> <path_train_ham> <path_test_spam> <path_test_ham>

3) SGDClassifier - Bag of words/ 4) Bern
	Usage - python <filename> <path_train_spam> <path_train_ham> <path_test_spam> <path_test_ham>

5 and 6) MCAP LR
	Usage - python <filename> <path_train_spam> <path_train_ham> <path_test_spam> <path_test_ham> <learning rate> <lambda>


the following imports are required:
import numpy as np
from os import listdir, getcwd
import re
from collections import Counter
import sys
import sklearn.metrics as skl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
