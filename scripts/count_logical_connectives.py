import re
import collections

def count_logical_connectives(file_path):
    # Expanded list of logical connectives and transition words
    connectives = [
        "therefore", "because", "so", "thus", "for", 
        "consequently", "hence", "accordingly", "since",
        "nevertheless", "however", "but", "although", "yet",
        "furthermore", "moreover", "additionally",
        "otherwise", "rather", "instead"
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Create a regex pattern to match whole words only
    results = {}
    for word in connectives:
        # \b ensures word boundaries
        pattern = rf'\b{word}\b'
        count = len(re.findall(pattern, text))
        results[word] = count

    # Sort results by frequency (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Connective':<15} | {'Count':<10}")
    print("-" * 28)
    for word, count in sorted_results:
        print(f"{word:<15} | {count:<10}")
    
    total = sum(results.values())
    print("-" * 28)
    print(f"{'TOTAL':<15} | {total:<10}")

if __name__ == "__main__":
    count_logical_connectives("../engine/nwt_corpus.txt")

