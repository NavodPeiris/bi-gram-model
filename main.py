import re
from nltk.tokenize import word_tokenize
from collections import defaultdict

corpus = "The quick brown fox jumps over the lazy dog. The lazy dog barked at the moon."

def data_preprocess(data):
    # Remove punctuation and special characters
    data = re.sub(r'[^\w\s]', '', data)

    # Convert to lowercase 
    data = data.lower()

    # tokenize the corpus
    token_words = word_tokenize(data)
    return token_words

tokens = data_preprocess(corpus)

# a dictionary of word counts
# each word that does not included will have a default value of 0
word_count = defaultdict(int)

# counting occurences of a word
for word in tokens:
    word_count[word] += 1

# counting occurences of each word pair
# a dictionary of bi-gram counts
# each bi-gram that does not included will have a default value of 0
# Count occurrences of each word pair (bi-gram)
bi_gram_count = defaultdict(int)
for i in range(len(tokens) - 1):
    bi_gram = (tokens[i], tokens[i + 1])
    bi_gram_count[bi_gram] += 1

bi_gram_probs = defaultdict(float)

V = len(word_count)  # Vocabulary size

# calculate bi-gram probabilities for all combinations of bi-grams using add 1 smoothing
for token in tokens:
    for tk in tokens:
        if token != tk:
            prev_word = token     # take one token as previous word
            next_word = tk         # take another as next word
            bi_gram = (prev_word, next_word)  # create a bi-gram
            count = bi_gram_count[bi_gram]    # get count of bi-gram occurrences (can be 0)
            # calculate probability (will not be zero as add 1 smoothing is applied)
            probability = (count + 1) / (word_count[prev_word] + V) 
            bi_gram_probs[bi_gram] = probability


def generate_text(seed_word, length, bi_gram_probabilities):
    generated_text = [seed_word]
    current_word = seed_word

    for i in range(length - 1):

        max_prob = 0
        next_word = ""

        # if current word is not in corpus, bi-gram probability for that word as previous word does not exist.
        # so calculate bi-gram probabilities using add 1 smoothing
        if current_word not in tokens:
            prev_word = current_word
            for word in tokens:
                next_word = word
                bi_gram = (prev_word, next_word)
                count = bi_gram_count[bi_gram]
                probability = (count + 1) / (word_count[prev_word] + V)
                bi_gram_probabilities[bi_gram] = probability

        for bi_gram, prob in bi_gram_probabilities.items():
            if bi_gram[0] == current_word:
                if prob > max_prob:
                    max_prob = prob
                    next_word = bi_gram[1]

        generated_text.append(next_word)
        current_word = next_word

    return ' '.join(generated_text)


# Generate text using the bi-gram model
generated_text = generate_text('barked', 10, bi_gram_probs)
print(generated_text)
