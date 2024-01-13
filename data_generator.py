import random
import json

# Generates a list of random words
def generate_random_words(num_words, min_length=1, max_length=7, alphabet="abcdefghijklmnopqrstuvwxyz"):
    word_list = set()

    while len(word_list) < num_words:
        word_length = random.randint(min_length, max_length)
        word = ''.join(random.choice(alphabet) for _ in range(word_length))
        word_list.add(word)

    return list(word_list)[:num_words]

# Reverses a list of words and returns a dictionary
def reverse_words(word_list):
    return {word: word[::-1] for word in word_list}

# Generates a JSONL string from a dictionary
def generate_jsonl(dictionary):
    jsonl = ""
    for key, value in dictionary.items():
        json_line = json.dumps({
            "prompt": key,
            "answer": value
        })
        jsonl += json_line + "\n"
    return jsonl

# Saves a JSONL string to a file
def save_jsonl(jsonl, filename):
    with open(filename, 'w') as file:
        file.write(jsonl)

if __name__ == "__main__":
    # Generate random words and reverse them
    word_list = generate_random_words(12000, 1, 4)
    reverse_word_dict = reverse_words(word_list)
    
    # Generate and save JSONL
    jsonl = generate_jsonl(reverse_word_dict)
    save_jsonl(jsonl, './data/output.jsonl')
