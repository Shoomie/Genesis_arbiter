from tokenizers import ByteLevelBPETokenizer
import os

def train_genesis_tokenizer(corpus_path, output_dir, vocab_size=8000):
    print(f"Initializing ByteLevelBPETokenizer with vocab_size={vocab_size}...")
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    print(f"Training on {corpus_path}...")
    tokenizer.train(files=[corpus_path], vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    # Save the trained tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving tokenizer to {output_dir}...")
    tokenizer.save_model(output_dir, "genesis_tokenizer")
    
    # Also save as a single json for easier use with transformers
    tokenizer.save(os.path.join(output_dir, "genesis_tokenizer.json"))
    print("Tokenizer training and saving complete.")

if __name__ == "__main__":
    corpus_file = "nwt_corpus.txt"
    output_directory = "."
    train_genesis_tokenizer(corpus_file, output_directory)
