import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('all')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer for text preprocessing
lemmatizer = WordNetLemmatizer()


# Load and preprocess intents data
def load_intents(json_file):
    with open(json_file) as file:
        data = json.load(file)
    return data


def preprocess(sentence):
    # Tokenize and lemmatize the sentence
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words


def prepare_training_data(intents):
    # Lists to store words, tags, and documents
    words, classes, documents = [], [], []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = preprocess(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    words = sorted(set(words))
    classes = sorted(set(classes))

    # Training data
    training_sentences = []
    training_labels = []

    for doc in documents:
        # Bag of words for each pattern
        bow = [1 if w in doc[0] else 0 for w in words]
        training_sentences.append(bow)
        # Output is a one-hot encoding for the class
        label = classes.index(doc[1])
        training_labels.append(label)

    return np.array(training_sentences), np.array(training_labels), words, classes


# Generate response based on input
def chatbot_response(intents, words, classes, input_sentence):
    input_bow = np.array([1 if w in preprocess(input_sentence) else 0 for w in words])

    max_similarity = 0
    best_response = "I don't understand. Could you rephrase?"

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_bow = np.array([1 if w in preprocess(pattern) else 0 for w in words])
            similarity = cosine_similarity([input_bow], [pattern_bow])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_response = random.choice(intent['responses'])

    return best_response


# Example usage
# Load intents (example intents.json file path)
intents = load_intents('intents.json')

# Prepare training data
training_sentences, training_labels, words, classes = prepare_training_data(intents)

# Run chatbot
print("Chatbot is ready! Type 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot_response(intents, words, classes, user_input)
    print("Bot:", response)
