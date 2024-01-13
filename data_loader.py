import json

# Load data from the JSONL file
def load_data_from_jsonl(jsonl_file):
    input_strings = []
    target_strings = []

    with open(jsonl_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        data = json.loads(line)
        input_strings.append(data['prompt'])  # Get the input from the JSONL
        target_strings.append(data['answer'])  # Get the target from the JSONL

    return input_strings, target_strings

if __name__ == "__main__":
    # Load data from the JSONL file
    jsonl_file = './data/output.jsonl'
    input_strings, target_strings = load_data_from_jsonl(jsonl_file)

    # Print the loaded data
    print("Input Strings:")
    for input_str in input_strings:
        print(input_str)

    print("\nTarget Strings:")
    for target_str in target_strings:
        print(target_str)