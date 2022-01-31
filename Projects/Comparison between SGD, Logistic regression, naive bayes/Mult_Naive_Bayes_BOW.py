from os import listdir, getcwd
import re
from collections import Counter
import numpy as np
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
        count_of_valid_words = Counter(valid_words)
        list_of_email_word_lists.append(count_of_valid_words)
    return vocabulary, list_of_email_word_lists


def create_vectors(list_of_vectors, vocabulary_list, spam=0):
    # for each email
    for i in range(len(list_of_vectors)):
        email_vector = list()
        # Loop through vocab list from spam and ham
        for word in vocabulary_list:
            if word in list_of_vectors[i].elements():
                email_vector.append(float(list_of_vectors[i][word]))
            else:
                email_vector.append(0)
        email_vector.append(spam)  # Since data is spam mark as spam
        if i != 0:
            temp = np.array(email_vector)
            temp = np.reshape(temp, (1, len(vocabulary_list) + 1))
            bow_array = np.append(bow_array, temp, axis=0)
        else:
            bow_array = np.array(email_vector)
            bow_array = np.reshape(bow_array, (1, len(vocabulary_list) + 1))
    return bow_array


def train_model(data):
    # all spam data => last column is True
    spam_data = data[data[:, -1] == 1, :-1]
    # count of each word in the spam data
    # Add one laplace smoothing is done using broadcasting
    count_of_words_in_spam = spam_data.sum(axis=0) + 1
    # Total number of words in spam
    count_of_all_words_in_spam = count_of_words_in_spam.sum()
    # probability of each word given email is spam
    list_of_p_word_spam = count_of_words_in_spam/count_of_all_words_in_spam
    # log probability
    list_of_log_p_word_spam = np.log(list_of_p_word_spam)

    # all ham data => last column is False
    ham_data = data[data[:, -1] == 0, :-1]
    # count of each word in the ham data
    # Add one laplace smoothing is done using broadcasting
    count_of_words_in_ham = ham_data.sum(axis=0) + 1
    # Total number of words in ham
    count_of_all_words_in_ham = count_of_words_in_ham.sum()
    # probability of each word given email is ham
    list_of_p_word_ham = count_of_words_in_ham/count_of_all_words_in_ham
    # log probability
    list_of_log_p_word_ham = np.log(list_of_p_word_ham)

    p_ham = len(ham_data)/len(data)
    p_spam = len(spam_data)/len(data)
    log_p_ham = np.log(p_ham)
    log_p_spam = np.log(p_spam)

    return list_of_log_p_word_spam, list_of_log_p_word_ham, log_p_spam, log_p_ham


def classify_data(data, p_word_spam, p_word_ham, p_spam, p_ham):
    # calculated Probabilities are 1 D arrays thus dot product with 2D array will result in 1 D array
    p_spam_test = p_spam + np.dot(data[:, :-1], p_word_spam)

    p_ham_test = p_ham + np.dot(data[:, :-1], p_word_ham)

    classification = np.argmax((p_ham_test, p_spam_test), 0)

    print(f'Accuracy of classifier: {skl.accuracy_score(classification, data[:, -1])}')
    print(f'Precision of classifier: {skl.precision_score(classification, data[:, -1])}')
    print(f'Recall of classifier: {skl.recall_score(classification, data[:, -1])}')
    print(f'F1 score of classifier: {skl.f1_score(classification, data[:, -1])}')


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
bow_spam_array = create_vectors(list_of_spam_word_lists, vocab_list, spam=1)

# Create ham vectors using vocab list
bow_ham_array = create_vectors(list_of_ham_word_lists, vocab_list)

# Append spam and ham data to create training dataset
final_bow_representation = np.append(bow_spam_array, bow_ham_array, axis=0)

# train model using final dataset
print('training the model....')
p_word_spam, p_word_ham, p_spam, p_ham = train_model(final_bow_representation)

# Get list of files in test spam path
list_of_spam_test_file_names = [f for f in listdir(spam_test_path)]
# Get list of files in test ham path
list_of_ham_test_file_names = [f for f in listdir(ham_test_path)]

# Create list of spam words for testing
_, list_of_test_spam_word = get_vocab_bow(spam_test_path, list_of_spam_test_file_names, vocab, test=1)

# Create list of ham words for testing
_, list_of_test_ham_word = get_vocab_bow(ham_test_path, list_of_ham_test_file_names, vocab, test=1)

print('Creating vectors for the test set....')
bow_test_spam_array = create_vectors(list_of_test_spam_word, vocab_list, spam=1)

bow_test_ham_array = create_vectors(list_of_test_ham_word, vocab_list)

test_data = np.append(bow_test_spam_array, bow_test_ham_array, axis=0)
print(f'shape of final test data: {test_data.shape}')

# test data is converted to Boolean array so that
# Dot product with log probabilities yields classification
test_data = test_data.astype(bool)

print('Classify the data....')
classify_data(test_data, p_word_spam, p_word_ham, p_spam, p_ham)

