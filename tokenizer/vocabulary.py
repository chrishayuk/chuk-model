# Define the vocabulary
#characters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.+=<>()-/*")
characters = list("0123456789,")
special_tokens = ["<pad>", "<eos>", "<cont>", "<endbatch>", "<data>","reverse:"]
vocabulary = special_tokens + characters

# Create token-ID mapping and reverse mapping
token_to_id = {token: id for id, token in enumerate(vocabulary)}

# Create ID-to-token mapping
id_to_token = {id: token for token, id in token_to_id.items()}

def print_vocabulary():
    print("Vocabulary:", vocabulary)
    print("Token to ID Mapping:", token_to_id)
