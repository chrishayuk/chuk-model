import json
import re
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Specify your dataset paths
original_dataset_path = './data/math/math_broad.jsonl'
preprocessed_dataset_path = './data/math/math_broad_preproc.jsonl'

# Initialize the SentencePiece BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Add individual digits as tokens
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for digit in digits:
    tokenizer.add_tokens([AddedToken(digit, normalized=False, special=False)])

# Trainer for the tokenizer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [original_dataset_path]

# Train the tokenizer
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("math_tokenizer.json")
