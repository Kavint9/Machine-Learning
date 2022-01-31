from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from os import listdir, getcwd
import re
from collections import Counter
import sys
import sklearn.metrics as skl

# list of stop words i.e. common words that should not impact the classification of a sentence
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
             'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
             'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
             'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
             'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
             "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
             "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
             "won't", 'wouldn', "wouldn't"]


def get_vocab_bow(path, file_list, vocabulary, test=0):
    list_of_email_word_lists = list()
    for i in range(len(file_list)):
        fp = open(path + '\\' + file_list[i], 'r', encoding='utf8', errors='ignore')
        email_text = str()
        for row in fp:
            email_text += row
        email_text.lower()
        email_text = re.sub(r'\W', ' ', email_text)     # replace non words with space => ![a-zA-Z_0-9]
        email_text = re.sub(r'\s+', ' ', email_text)    # replace white-space characters => space tab newline etc.
                                                        # + implies removal of one or more
        email_text = re.sub(r'\d', ' ', email_text)     # replace numbers with space
        words_in_email = email_text.split()
        valid_words = [word for word in words_in_email if word not in stopwords]
        if test == 0:
            vocabulary.update(valid_words)
        set_valid_words = set(valid_words)
        list_of_email_word_lists.append(set_valid_words)
    return vocabulary, list_of_email_word_lists


def create_vectors(list_of_vectors, vocabulary_list, spam=0):
    # for each email
    for i in range(len(list_of_vectors)):
        email_vector = list()
        # Loop through vocab list from spam and ham
        for word in vocabulary_list:
            if word in list_of_vectors[i]:
                email_vector.append(1)
            else:
                email_vector.append(0)
        email_vector.append(spam)  # If data is spam mark as spam
        if i != 0:
            temp = np.array(email_vector)
            temp = np.reshape(temp, (1, len(vocabulary_list) + 1))
            bow_array = np.append(bow_array, temp, axis=0)
        else:
            bow_array = np.array(email_vector)
            bow_array = np.reshape(bow_array, (1, len(vocabulary_list) + 1))
    return bow_array


spam_path = getcwd() + str(sys.argv[1])
ham_path = getcwd() + str(sys.argv[2])
spam_test_path = getcwd() + str(sys.argv[3])
ham_test_path = getcwd() + str(sys.argv[4])

list_of_spam_file_names = [f for f in listdir(spam_path)]
list_of_ham_file_names = [f for f in listdir(ham_path)]
vocab = Counter()
vocab, list_of_spam_word_lists = get_vocab_bow(spam_path, list_of_spam_file_names, vocab)
vocab, list_of_ham_word_lists = get_vocab_bow(ham_path, list_of_ham_file_names, vocab)
print(f'vocab formed using spam and ham data from training data {len(vocab)}')

vocab_list = [el for el, count in vocab.items() if count > 1]
print(f'vocab list after feature selection to avoid overfitting {len(vocab_list)}')

print('Creating spam and ham vectors for training data....')
# Create spam array using vocab list
bern_spam_array = create_vectors(list_of_spam_word_lists, vocab_list, spam=1)

# Create ham vectors using vocab list
bern_ham_array = create_vectors(list_of_ham_word_lists, vocab_list)

# Append spam and ham data to create training dataset
final_bern_representation = np.append(bern_spam_array, bern_ham_array, axis=0)

np.random.shuffle(final_bern_representation)
print(f'Shape of training data: {final_bern_representation.shape}')

# Identify best parameters using GridSearchCV
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
}

model = SGDClassifier(max_iter=1000)
clf = GridSearchCV(model, param_grid=params)

X = final_bern_representation[:, :-1]
y = final_bern_representation[:, -1]

print('Identifying best parameters using Grid Search....')
clf.fit(X, y)

print(f'Best params for SGDClassifier: {clf.best_params_}')

# Get list of files in test spam path
list_of_spam_test_file_names = [f for f in listdir(spam_test_path)]
# Get list of files in test ham path
list_of_ham_test_file_names = [f for f in listdir(ham_test_path)]

# Create list of spam words for testing
_, list_of_test_spam_word = get_vocab_bow(spam_test_path, list_of_spam_test_file_names, vocab, test=1)

# Create list of ham words for testing
_, list_of_test_ham_word = get_vocab_bow(ham_test_path, list_of_ham_test_file_names, vocab, test=1)

print('Creating vectors for the test set....')
bern_test_spam_array = create_vectors(list_of_test_spam_word, vocab_list, spam=1)

bern_test_ham_array = create_vectors(list_of_test_ham_word, vocab_list)

test_data = np.append(bern_test_spam_array, bern_test_ham_array, axis=0)
print(f'shape of final test data: {test_data.shape}')

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

yhat_test = clf.predict(X_test)

print(f'Accuracy of classifier: {skl.accuracy_score(yhat_test, y_test)}')
print(f'Precision of classifier: {skl.precision_score(yhat_test, y_test)}')
print(f'Recall of classifier: {skl.recall_score(yhat_test, y_test)}')
print(f'F1 score of classifier: {skl.f1_score(yhat_test, y_test)}')



