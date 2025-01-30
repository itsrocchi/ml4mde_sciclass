# Step 1: Load and preprocess the data
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import kagglehub
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import nltk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import recall_score



# Download latest version of the dataset
path = kagglehub.dataset_download("shivanandmn/multilabel-classification-dataset")

print("Path to dataset files:", path)

# Load the dataset
url_train = f'{path}/train.csv'
train_set  = pd.read_csv(url_train)

train_set.drop(columns=['ID'], inplace=True)

# Combine the category columns into a list of categories for each paper
categories = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
train_set['Categories'] = train_set[categories].apply(lambda row: [cat for cat in categories if row[cat] == 1], axis=1)

# Flatten the list of categories
flattened_categories = train_set['Categories'].explode()

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(flattened_categories), y=flattened_categories)

# Create a dictionary mapping each category to its weight
class_weight_dict = {category: weight for category, weight in zip(np.unique(flattened_categories), class_weights)}

print("Class weights:", class_weight_dict)

# Apply class weights to the classifier
classifier_balanced = LogisticRegression(class_weight=class_weight_dict)

contractions_dict = {
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "isn't": "is not",
    "won't": "will not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "it's": "it is",
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "I won't": "I will not",
    "you won't": "you will not",
    "he won't": "he will not",
    "she won't": "she will not",
    "we won't": "we will not",
    "they won't": "they will not",
}


def expand_contractions(text, contractions_dict):
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text


# Preprocess the text data
train_set['TITLE'] = train_set['TITLE'].str.lower().str.replace(r'[^\\w\\s]', '').str.replace(r'\\d+', '').str.strip()
train_set['ABSTRACT'] = train_set['ABSTRACT'].str.lower().str.replace(r'[^\\w\\s]', '').str.replace(r'\\d+', '').str.strip()
train_set['PROCESSED_TEXT'] = train_set['TITLE'] + ' ' + train_set['ABSTRACT']

# Scarica il set di stop words in italiano o ingles
#nltk.download('stopwords')

# Stop words in inglese
stop_words = set(nltk.corpus.stopwords.words('english'))

# Funzione per rimuovere stop words
def remove_stop_words(text):
    # Tokenizza il testo, rimuove stop words e parole corte
    words = " ".join([contractions_dict.get(word, word) for word in text.split()])
    words = [word for word in text.split() if word not in stop_words and len(word) > 1]
    
    return " ".join(words)

# Applica la funzione alla colonna 'text' o 'ABSTRACT'
train_set['PROCESSED_TEXT'] = train_set['PROCESSED_TEXT'].apply(lambda x: expand_contractions(x, contractions_dict))
train_set['PROCESSED_TEXT'] = train_set['PROCESSED_TEXT'].apply(remove_stop_words)

# Binarize the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_set['Categories'])

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(train_set['TEXT'])

# Step 2: Implement a machine learning classifier
classifier = LogisticRegression()

# Step 3: Evaluate the classifier using suitable metrics
kf = KFold(n_splits=5, shuffle=True)
f1_scores = []
accuracy_scores = []

ovr_classifier = OneVsRestClassifier(classifier)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ovr_classifier.fit(X_train, y_train)
    y_pred = ovr_classifier.predict(X_test)
    
    f1_scores.append(f1_score(y_test, y_pred, average='micro'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))


# Step 4: Split data for k-fold cross-validation
# (Already done in the loop above)

# Step 5: Run the evaluation
print("F1 Scores: ", f1_scores)
print("Accuracy Scores: ", accuracy_scores)

# Step 6: Compare with dedicated baselines
# (Assuming we have baseline scores to compare with)

# Step 7: Analyze the obtained results
print("Average F1 Score: ", sum(f1_scores) / len(f1_scores))
print("Average Accuracy Score: ", sum(accuracy_scores) / len(accuracy_scores))

# Scarica risorse necessarie
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

# Funzione per ottenere il tag di WordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

# Funzione per lemmatizzare il testo
def lemmatize_text(text):
    words = text.split()
    pos_tags = pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized_words)

# Applica la lemmatizzazione alla colonna Processed_Text
train_set['LEMMATIZED_TEXT'] = train_set['PROCESSED_TEXT'].apply(lemmatize_text)

# Visualizza il risultato
print(train_set[['PROCESSED_TEXT', 'LEMMATIZED_TEXT']])

# Binarize the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_set['Categories'])

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(train_set['LEMMATIZED_TEXT'])
# Step 2: Implement a machine learning classifier
classifier = LogisticRegression()

# Step 3: Evaluate the classifier using suitable metrics
kf = KFold(n_splits=5, shuffle=True)
f1_scores = []
accuracy_scores = []
recall_scores = []

ovr_classifier = OneVsRestClassifier(classifier)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ovr_classifier.fit(X_train, y_train)
    y_pred = ovr_classifier.predict(X_test)
    
    f1_scores.append(f1_score(y_test, y_pred, average='micro'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred, average='micro'))

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import recall_score

print("Accuracy Scores: ", accuracy_scores)

# Step 6: Compare with dedicated baselines
# (Assuming we have baseline scores to compare with)

# Step 7: Analyze the obtained results
print("Average F1 Score: ", sum(f1_scores) / len(f1_scores))
print("Average Accuracy Score: ", sum(accuracy_scores) / len(accuracy_scores))
print("Average Recall Score: ", sum(recall_scores) / len(recall_scores))

# Select the columns 'TEXT' and 'PROCESSED_TEXT'
selected_columns = train_set[['TEXT', 'LEMMATIZED_TEXT']]

# Save the selected columns to a CSV file with each column in a separate column
selected_columns.to_csv('selected_columns1.csv', index=False)

def repreprocess_text(text):
    # Remove formulas
    text = re.sub(r'\$.*?\$', '', text)
    
    return text

train_set['REPROCESSED_TEXT'] = train_set['LEMMATIZED_TEXT'].apply(repreprocess_text)

# Binarize the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_set['Categories'])

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(train_set['REPROCESSED_TEXT'])

# Step 3: Evaluate the classifier using suitable metrics
kf = KFold(n_splits=5, shuffle=True)
f1_scores = []
accuracy_scores = []
recall_scores = []

ovr_classifier = OneVsRestClassifier(classifier)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ovr_classifier.fit(X_train, y_train)
    y_pred = ovr_classifier.predict(X_test)
    
    f1_scores.append(f1_score(y_test, y_pred, average='micro'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred, average='micro'))

# Step 4: Split data for k-fold cross-validation
# (Already done in the loop above)

# Step 5: Run the evaluation
print("F1 Scores: ", f1_scores)
print("Recall Scores: ", recall_scores)

print("Accuracy Scores: ", accuracy_scores)

# Step 7: Analyze the obtained results
print("Average F1 Score: ", sum(f1_scores) / len(f1_scores))
print("Average Accuracy Score: ", sum(accuracy_scores) / len(accuracy_scores))
print("Average Recall Score: ", sum(recall_scores) / len(recall_scores))