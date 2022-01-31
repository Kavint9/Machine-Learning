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
        set_valid_words = Counter(valid_words)
        list_of_email_word_lists.append(set_valid_words)
    return vocabulary, list_of_email_word_lists


def create_vectors(list_of_vectors, vocabulary_list, spam=0):
    # for each email
    for i in range(len(list_of_vectors)):
        email_vector = list()
        # Loop through vocab list from spam and ham
        for word in vocabulary_list:
            if word in list_of_vectors[i]:
                email_vector.append(float(list_of_vectors[i][word]))
            else:
                email_vector.append(float(0))
        email_vector.append(spam)  # If data is spam mark as spam
        if i != 0:
            temp = np.array(email_vector)
            temp = np.reshape(temp, (1, len(vocabulary_list) + 1))
            bow_array = np.append(bow_array, temp, axis=0)
        else:
            bow_array = np.array(email_vector)
            bow_array = np.reshape(bow_array, (1, len(vocabulary_list) + 1))
    return bow_array


def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s


def initialize_weights(dim):
    w = np.zeros((dim, 1))
    w0 = 0
    return w, w0


def propagate(w, w0, penalty, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + w0)
    cost = -(1 / m) * np.sum(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))  # - (penalty * w))
    dw = (1 / m) * (np.dot(X, (A - Y).T) + (penalty * w))
    dw0 = (1 / m) * (np.sum(A - Y) + (penalty * w0))
    grads = {"dw": dw,
             "dw0": dw0}

    return (dw, dw0), cost


def optimize(w, w0, X, Y, num_iterations, learning_rate, penalty=0, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, w0, penalty, X, Y)

        # Retrieve derivatives from grads
        dw = grads[0]
        dw0 = grads[1]
        #         print(dw)
        w = w - (learning_rate * dw)  # +  (learning_rate * penalty * w)
        w0 = w0 - (learning_rate * dw0)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "w0": w0}

    grads = {"dw": dw,
             "dw0": dw0}

    return params, grads, costs


def predict(w, w0, X):
    m=X.shape[1]
    A = sigmoid(np.dot(w.T, X) + w0)
    classification = (A > 0.5).astype(int)
    return classification


spam_path = getcwd() + str(sys.argv[1])
ham_path = getcwd() + str(sys.argv[2])
spam_test_path = getcwd() + str(sys.argv[3])
ham_test_path = getcwd() + str(sys.argv[4])
learning_rate = float(sys.argv[5])
penalty = float(sys.argv[6])


list_of_spam_file_names = [f for f in listdir(spam_path)]
list_of_ham_file_names = [f for f in listdir(ham_path)]
vocab = Counter()

vocab, list_of_spam_word_lists = get_vocab_bow(spam_path, list_of_spam_file_names, vocab)
vocab, list_of_ham_word_lists = get_vocab_bow(ham_path, list_of_ham_file_names, vocab)
print(f'vocab formed using spam and ham data from training data {len(vocab)}')
vocab_list = vocab

print('Creating spam and ham vectors for training data....')
# Create spam array using vocab list
bow_spam_array = create_vectors(list_of_spam_word_lists, vocab_list, spam=1)

# Create ham vectors using vocab list
bow_ham_array = create_vectors(list_of_ham_word_lists, vocab_list)

# Append spam and ham data to create training dataset
final_bow_representation = np.append(bow_spam_array, bow_ham_array, axis=0)

np.random.shuffle(final_bow_representation)
print(f'Shape of training data: {final_bow_representation.shape}')

train_data = final_bow_representation[:int(0.7*len(final_bow_representation)), :]
valid_data = final_bow_representation[int(0.7*len(final_bow_representation)):, :]
X_train = train_data[:,:-1].T
Y_train = train_data[:, -1].T
X_valid = valid_data[:,:-1].T
Y_valid = valid_data[:, -1].T

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

X_final = final_bow_representation[:, :-1].T
y_final = final_bow_representation[:, -1].T

feature_count = X_train.shape[0]
w, w0 = initialize_weights(feature_count)
print("Optimize weights using training set....")
parameters, grads, costs = optimize(w, w0, X_train, Y_train, 1000, learning_rate=learning_rate, penalty=penalty, print_cost=False)

classification = predict(parameters["w"], parameters["w0"], X_valid)
classification = classification.flatten()
print(f'Accuracy of validation set: {skl.accuracy_score(classification, Y_valid)}')
print(f'Precision of validation set: {skl.precision_score(classification, Y_valid)}')
print(f'Recall of validation set: {skl.recall_score(classification, Y_valid)}')
print(f'F1 score of validation set: {skl.f1_score(classification, Y_valid)}')

print("Optimize weights using training & validation set....")
feature_count = X_train.shape[0]
w, w0 = initialize_weights(feature_count)
parameters, grads, costs = optimize(w, w0, X_final, y_final, 1000, learning_rate=learning_rate, penalty=penalty, print_cost=False)

X_test = test_data[:, :-1].T
print(X_test.shape)
Y_test = test_data[:, -1].T
print(Y_test.shape)

predictions = predict(parameters["w"], parameters["w0"], X_test)
predictions = predictions.flatten()
print(f'Accuracy of test set: {skl.accuracy_score(predictions, Y_test)}')
print(f'Precision of test set: {skl.precision_score(predictions, Y_test)}')
print(f'Recall of test set: {skl.recall_score(predictions, Y_test)}')
print(f'F1 score of test set: {skl.f1_score(predictions, Y_test)}')


