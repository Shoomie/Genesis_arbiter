import os
import sys
from models.tokenizer import GenesisTokenizer

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def count_tokenizer_statistics(corpus_path, tokenizer_path):
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file {corpus_path} not found.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer file {tokenizer_path} not found.")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = GenesisTokenizer(tokenizer_path)

    print(f"Reading corpus {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Encoding corpus (this may take a moment)...")
    # Using the underlying tokenizer's encode for detailed access
    encoding = tokenizer.tokenizer.encode(text)
    token_ids = encoding.ids
    token_strings = encoding.tokens

    # Analysis
    unique_token_ids = set(token_ids)
    vocab_size = tokenizer.vocab_size

    print(f"\n--- Tokenizer Coherency Statistics ---")
    print(f"Total tokens in corpus: {len(token_ids):,}")
    print(f"Unique tokens used: {len(unique_token_ids):,}")
    print(f"Model Vocabulary Size: {vocab_size:,}")
    print(f"Vocab Utilization: {(len(unique_token_ids)/vocab_size)*100:.2f}%")

    # Specifically check for "Jehovah"
    jhvh_id = tokenizer.tokenizer.token_to_id("Jehovah")
    if jhvh_id:
        count = token_ids.count(jhvh_id)
        print(f"\n'Jehovah' Token (ID {jhvh_id}) count: {count:,}")
    
    # Analyze subword fragmentation (optional but helpful for coherency)
    # Words are often preceded by 'Ġ' in ByteLevelBPE
    print(f"\nExample tokens from corpus path:")
    print(token_strings[100:120])

if __name__ == "__main__":
    corpus_file = "nwt_corpus.txt"
    tokenizer_file = "genesis_tokenizer.json"
    count_tokenizer_statistics(corpus_file, tokenizer_file)
