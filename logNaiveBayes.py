#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:33:45 2024

@author: ganeshreddypuli
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

dataframe_emails = pd.read_csv('emails.csv')
dataframe_emails.head()

no_of_spam_emails = dataframe_emails.spam.sum()
no_of_ham_emails = len(dataframe_emails) - dataframe_emails.spam.sum()

print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {no_of_spam_emails/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {no_of_ham_emails/len(dataframe_emails):.4f}")

def preprocess_emails(df):

    # Shuffles the dataset
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    # Removes the "Subject:" string, which comprises the first 9 characters of each email. Also, convert it to a numpy array.
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # Convert the labels to numpy array
    Y = df.spam.to_numpy()
    return X, Y

X, Y = preprocess_emails(dataframe_emails)

def preprocess_text(X):

    # Make a set with the stopwords and punctuation
    stop = set(stopwords.words('english') + list(string.punctuation))

    # The next lines will handle the case where a single email is passed instead of an array of emails.
    if isinstance(X, str):
        X = np.array([X])

    # The result will be stored in a list
    X_preprocessed = []

    for i, email in enumerate(X):
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)
        
    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed 

X_treated = preprocess_text(X)

'''email_index = 989
print(f"Email before preprocessing: {X[email_index]}")
print(f"Email after preprocessing: {X_treated[email_index]}")'''

TRAIN_SIZE = int(0.80*len(X_treated)) # 80% of the samples will be used to train.

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

print(f"\nProportion of spam in train dataset: {sum(Y_train == 1)/len(Y_train):.4f}")
print(f"Proportion of spam in test dataset: {sum(Y_test == 1)/len(Y_test):.4f}")

def get_word_frequency(X,Y):

    # Creates an empty dictionary
    word_dict = {}

    num_emails = len(X)

    # Iterates over every processed email and its label
    for i in range(num_emails):
        # Get the i-th email
        email = X[i] 
        # Get the i-th label. This indicates whether the email is spam or not. 1 = spam and 0 = ham
        # The variable name cls is an abbreviation for class, a reserved word in Python.
        cls = Y[i] 
        # To avoid counting the same word twice in an email, remove duplicates by casting the email as a set
        email = set(email) 
        # Iterates over every distinct word in the email
        for word in email:
            # If the word is not already in the dictionary, manually add it. Remember that you will start every word count as 1 both in spam and ham
            if word not in word_dict.keys():
                word_dict[word] = {"spam": 1, "ham": 1}
            # Add one occurrence for that specific word in the key ham if cls == 0 and spam if cls == 1. 
            if cls == 0:    
                word_dict[word]["ham"] += 1
            if cls == 1:
                word_dict[word]["spam"] += 1

    return word_dict

word_frequency = get_word_frequency(X_train,Y_train)
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}
print("\nClass frequency in train dataset: ",class_frequency)

def prob_word_given_class(word, cls, word_frequency, class_frequency):

    # Get the amount of times the word appears with the given class (class is stores in spam variable)
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class  

def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    
    # prob starts at 0 because it will be updated by summing it with the current log(P(word | class)) in every iteration
    prob = 0

    for word in treated_email: 
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys(): 
            # Update the prob by summing it with log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls, word_frequency, class_frequency))

    return prob

def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    

    log_prob_email_given_spam = log_prob_email_given_class(treated_email, "spam", word_frequency, class_frequency)

    log_prob_email_given_ham = log_prob_email_given_class(treated_email, "ham", word_frequency, class_frequency)

    p_spam = class_frequency["spam"] / (class_frequency["spam"] + class_frequency["ham"])

    p_ham = class_frequency["ham"] / (class_frequency["spam"] + class_frequency["ham"])

    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam

    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham

    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)
    
    # Compares both values and choose the class corresponding to the higher value
    elif log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0

print("\nExample emails:")    
example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)

print(f"\nEmail: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {log_naive_bayes(treated_email, word_frequency, class_frequency)}")
example_email = "Our meeting will happen in the main office. Please be there in time."
treated_email = preprocess_text(example_email)

print(f"\nEmail: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {log_naive_bayes(treated_email, word_frequency, class_frequency)}")

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Let's get the predictions for the test set:
# Create an empty list to store the predictions
Y_pred = []

# Iterate over every email in the test set
for i, email in enumerate(X_test):
    # Perform prediction
    prediction = log_naive_bayes(email, word_frequency = word_frequency, class_frequency = class_frequency)
    # Add it to the list
    if (Y_test[i] == 1 and prediction == 1):
        true_positives += 1
    elif (Y_test[i] == 0 and prediction == 0):
        true_negatives += 1
    elif (Y_test[i] == 0 and prediction == 1):
        false_positives += 1
    elif (Y_test[i] == 1 and prediction == 0):
        false_negatives += 1
    Y_pred.append(prediction)

print("\nMetrics calculation:")    
print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")
print(f"The number of false positives is: {false_positives}\nThe number of false negatives is: {false_negatives}")
    
accuracy = (true_positives + true_negatives)/len(Y_test)
print(f"\nAccuracy is: {accuracy:.4f}")
recall = true_positives/(true_positives + false_negatives)
print(f"Recall is: {recall:.4f}")
precision = true_positives/(true_positives + false_positives)
print(f"Precision is: {precision:.4f}")

print("\nInterpretation of metrics:")
print("-A recall of 98% states that out of 100 emails which are actually spam, 98 are being classified as spam by our model and the remaining 2 are classified as ham which is ok.")
print("-A precision of 98% states that out of 100 emails which are predicted as spam by our model, 98 are actually spam and the remaining 2 are ham which are being misclassified as spam by our model which is again ok.")
print("-Pretty good model. Yay!")


