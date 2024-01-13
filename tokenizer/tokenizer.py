from .vocabulary import token_to_id, id_to_token, print_vocabulary

# encodes the string to a token
def encode(input_string):
    token_ids = []
    
    # Check if the input_string starts with 'reverse:'
    if input_string.startswith('reverse:'):
        # Split into parts
        parts = input_string.split('reverse:')
        
        # Add the token ID for 'reverse:'
        token_ids.append(token_to_id['reverse:'])

        # Encoding the content part
        content_tokens = parts[1] if len(parts) > 1 else ""
        token_ids.extend([token_to_id[char] for char in content_tokens])

        # Add the token ID for the <eos> token
        token_ids.append(token_to_id["<eos>"])
    else:
        # For regular strings, encode each character individually
        token_ids.extend([token_to_id[char] for char in input_string])

    return token_ids

# decodes the tokens back to a string
def decode(token_ids):
    # loops through each token id, and converts back to a string, then joins together
    return ''.join([id_to_token[id] for id in token_ids if id_to_token[id] not in ("<eos>", "<pad>")])

def get_vocab_size():
    # returns the size of the vocab
    return len(token_to_id)