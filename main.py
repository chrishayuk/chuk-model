from tokenizer.vocabulary import print_vocabulary
from tokenizer.tokenizer import encode, decode, get_vocab_size
from model.model_config_manager import ModelConfigManager
from model.model_loader import load_model

print_vocabulary()
print("Vocabulary Length:", get_vocab_size())

# performs a sample encoding and decoding
input_string = "reverse:example123"
tokenized_input = encode(input_string)
print("Encoded:", tokenized_input)
print("Decoded:", decode(tokenized_input))

# Load the model configuration
model_config_manager = ModelConfigManager("./")
model_config = model_config_manager.load_config()
model_config.vocab_size = get_vocab_size()
print(f"Model Config: {model_config}")

# Load the model
model = load_model(model_config)
print(f"Model: {model}")
print(f"Model is on device: {next(model.parameters()).device}")