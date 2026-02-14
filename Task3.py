import nltk
nltk.data.path.append('C:/Users/Asus/PycharmProjects/pranavR/tasks/.venv/nltk_data')
nltk.download('punkt_tab')
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load knowledge base
with open("knowledge.txt", "r", encoding="utf-8") as file:
    raw_text = file.read().lower()

# Tokenize sentences
sentence_tokens = nltk.sent_tokenize(raw_text)

# Greeting inputs and responses
greeting_inputs = ("hello", "hi", "hey", "good morning", "good evening")
greeting_responses = ["Hello!", "Hi there!", "Hey!", "Greetings!"]

def greeting(user_input):
    for word in user_input.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None

# Text preprocessing
def preprocess(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))

# Chatbot response logic
def generate_response(user_input):
    user_input = preprocess(user_input)
    sentence_tokens.append(user_input)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentence_tokens)

    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    index = similarity.argsort()[0][-2]

    similarity_scores = similarity.flatten()
    similarity_scores.sort()
    score = similarity_scores[-2]

    sentence_tokens.pop()

    if score == 0:
        return "Sorry, I don't understand that."
    else:
        return sentence_tokens[index]

# Chat loop
print("Chatbot: Hello! Ask me a question. Type 'bye' to exit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    if greeting(user_input) is not None:
        print("Chatbot:", greeting(user_input))
    else:
        print("Chatbot:", generate_response(user_input))
