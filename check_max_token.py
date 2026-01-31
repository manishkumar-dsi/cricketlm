from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', cache_dir='./my_local_cache/', local_files_only=True)
data_dir = Path('output2')          # <-- make sure this folder exists

max_len = 0
for jsonl_path in data_dir.glob('*.jsonl'):     # returns Path objects, no wildcard left
    with jsonl_path.open(encoding='utf-8') as f:
        for line in f:
            tokens = tokenizer.encode(line.strip())
            max_len = max(max_len, len(tokens))
        print(f"Checking: {jsonl_path} and max length: {max_len}")    

print("Longest input (tokens):", max_len)
