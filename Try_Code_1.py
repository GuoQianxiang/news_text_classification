from nltk.corpus import reuters
import spacy
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
nlp = spacy.load("en_core_web_md")


def tokenize(text):
    min_length = 3
    tokens = [word.lemma_ for word in nlp(text) if not word.is_stop]
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
    return filtered_tokens

def represent_tfidf(train_docs, test_docs):
    representer = TfidfVectorizer(tokenizer=tokenize)
    # Learn and transform train documents
    vectorised_train_documents = representer.fit_transform(train_docs)
    vectorised_test_documents = representer.transform(test_docs)
    return vectorised_train_documents, vectorised_test_documents

def doc2vec(text):
    min_length = 3
    p = re.compile('[a-zA-Z]+')
    tokens = [token for token in nlp(text) if not token.is_stop and
              p.match(token.text) and
              len(token.text) >= min_length]
    doc = np.average([token.vector for token in tokens], axis=0)
    return doc

def represent_doc2vec(train_docs, test_docs):
    vectorised_train_documents = [doc2vec(doc) for doc in train_docs]
    vectorised_test_documents = [doc2vec(doc) for doc in test_docs]
    return vectorised_train_documents, vectorised_test_documents

def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))