"""
Test script for weighted masking implementation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

from datasets.bible_weighted_masking import WeightedMaskingDataset
from models.tokenizer import GenesisTokenizer

def test_weighted_masking():
    print("=" * 70)
    print("Testing Weighted Masking Implementation")
    print("=" * 70)
    print()
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'engine', 'genesis_tokenizer.json')
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'engine', 'nwt_corpus.txt')
    
    if not os.path.exists(tokenizer_path):
        print(f"[ERROR] Tokenizer not found at {tokenizer_path}")
        return
    
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus not found at {corpus_path}")
        return
    
    print(f"[OK] Loading tokenizer from {tokenizer_path}")
    tokenizer = GenesisTokenizer(tokenizer_path)
    print(f"[OK] Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    print()
    
    # Create dataset
    print("Creating WeightedMaskingDataset...")
    dataset = WeightedMaskingDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_len=512,
        base_prob=0.4
    )
    print()
    
    # Test a few samples
    print("Testing sample retrieval...")
    for i in range(3):
        x, y = dataset[i]
        
        # Count masked tokens
        mask_id = dataset.mask_token_id
        masked_count = (x == mask_id).sum().item()
        
        # Count connectives in labels
        connective_count = sum(1 for token in y.tolist() if token in dataset.token_weights)
        
        print(f"Sample {i}:")
        print(f"  Sequence length: {len(x)}")
        print(f"  Masked tokens: {masked_count}")
        print(f"  Connectives in labels: {connective_count}")
        
        # Show a few masked positions
        if masked_count > 0:
            masked_positions = (x == mask_id).nonzero(as_tuple=True)[0][:5]
            print(f"  First masked positions: {masked_positions.tolist()}")
            print(f"  Corresponding labels: {[tokenizer.tokenizer.decode([y[pos].item()]) for pos in masked_positions]}")
        print()
    
    print("=" * 70)
    print("[SUCCESS] Weighted masking test completed successfully!")
    print("=" * 70)
    print()
    print("To enable in training:")
    print("  1. Edit engine/train_configs/high_vram.toml")
    print("  2. Set [masking] enabled = true")
    print("  3. Run: python engine/train.py")
    print()

if __name__ == "__main__":
    test_weighted_masking()
