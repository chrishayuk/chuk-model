import torch
import argparse
from model.transformer_model import TransformerModel
from tokenizer.tokenizer import encode, decode, token_to_id

# loads the model
def load_model(model_path, model_params):
    # initialize the transformer model
    model = TransformerModel(**model_params)

    # load the weights, for now use CPU
    # in the future need to update for CUDA, mps etc
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # put the model in eval mode
    model.eval()

    # return the model
    return model

# pre-processes the prompt, i.e. tokenizes, and shoves into a tensor
def preprocess_input(text, tokenizer, max_seq_length):
    # Tokenize the text
    tokenized_text = tokenizer(text)
    print(tokenized_text)

    # Ensure the sequence is within the max sequence length
    if len(tokenized_text) > max_seq_length:
        tokenized_text = tokenized_text[:max_seq_length]

    # Convert to tensor
    input_tensor = torch.tensor(tokenized_text).unsqueeze(0)

    # return the tensor
    return input_tensor

# post-processes the output, i.e. extracts it, and decodes the tokens
def process_output(output):
    # Convert the output tensor to a list of token IDs
    token_ids = output.argmax(dim=-1).tolist()[0] 
    print(token_ids)

    # Decode the token IDs back to a string
    decoded_output = decode(token_ids)

    # return the output
    return decoded_output

def predict(text, model, tokenizer, max_seq_length):
    # Preprocess the input text
    input_tensor = preprocess_input(text, tokenizer, max_seq_length)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Debugging: Print the logits for the first token
    #first_token_logits = output[0, 0, :]  # Assuming output shape is [batch, seq_length, vocab_size]
    #print("First token logits:", first_token_logits)
    #print("Softmax probabilities:", torch.softmax(first_token_logits, dim=-1))

    # post-processes the output, i.e. converts the tokens back to a string
    return process_output(output)

if __name__ == "__main__":
    # set up the parser
    parser = argparse.ArgumentParser(description="Predict a reversed string using a Transformer model.")

    # accept a single argument
    parser.add_argument("prompt", type=str, help="The text to be reversed")

    # parse the args
    args = parser.parse_args()

    # Model parameters and path
    model_path = './saved_models/trained_model_final.pth'

    # setup model params
    model_params = {
        'vocab_size': len(token_to_id),
        'd_model': 512,
        'nhead': 8,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'max_seq_length': 200,
    }

    # Load the model
    model = load_model(model_path, model_params)

    # Make a prediction
    #input_sequence = [int(num) for num in args.prompt.split(',')]
    #prediction = predict(args.prompt, model, encode, model_params['max_seq_length'])
    prediction = predict(args.prompt, model, encode, model_params['max_seq_length'])

    # show the prediction
    print("Prediction:", prediction)
