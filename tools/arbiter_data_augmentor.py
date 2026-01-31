"""
Arbiter Data Augmentor: Synthetic Reasoning Trace Generation

This module generates reasoning traces and synthetic data from the NWT corpus
to maximize data utilization in the data-constrained regime.

Augmentation Strategies:
1. Question-Answer Pairs: Transform declarative → interrogative
2. Cross-Reference Chains: Link verses via metadata for transitivity
3. Adversarial Contradictions: Generate theologically inconsistent statements
4. Genealogical Graph Traversal: Auto-generate kinship reasoning

Output: JSONL format compatible with torchtitan training loops
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import random
from collections import defaultdict


@dataclass
class ReasoningTrace:
    """A single reasoning trace entry."""
    input: str
    output: str
    reasoning_type: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ArbiterDataAugmentor:
    """
    Synthetic data generator for theological reasoning.
    
    Usage:
        augmentor = ArbiterDataAugmentor(corpus_path="./nwt_corpus.txt")
        traces = augmentor.generate_all()
        augmentor.save_jsonl(traces, "./augmented_data.jsonl")
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        
        # Load corpus
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.corpus_text = f.read()
        
        # Parse into verses (simplified)
        self.verses = self._parse_verses()
        
        print(f"[DataAugmentor] Loaded {len(self.verses)} verses from corpus")
    
    def _parse_verses(self) -> List[Dict[str, str]]:
        """
        Parse corpus into verse structure.
        
        Simplified parser assuming format: "Book Chapter:Verse Text"
        Adjust based on actual NWT corpus format.
        """
        verses = []
        
        # Pattern: Genesis 1:1 In the beginning...
        pattern = r'([A-Z][a-z]+\s+\d+:\d+)\s+(.+?)(?=\n[A-Z][a-z]+\s+\d+:\d+|\Z)'
        
        matches = re.finditer(pattern, self.corpus_text, re.DOTALL)
        
        for match in matches:
            reference = match.group(1).strip()
            text = match.group(2).strip()
            
            verses.append({
                'reference': reference,
                'text': text
            })
        
        # Fallback: if parsing fails, create simple chunks
        if not verses:
            chunks = self.corpus_text.split('\n\n')
            for i, chunk in enumerate(chunks[:1000]):  # Limit for testing
                verses.append({
                    'reference': f'Verse {i+1}',
                    'text': chunk.strip()
                })
        
        return verses
    
    def generate_qa_pairs(self, num_samples: int = 500) -> List[ReasoningTrace]:
        """
        Strategy 1: Question-Answer Pairs
        
        Transform: "Jesus went to Jerusalem" 
        Into Q: "Where did Jesus go?" A: "Jerusalem"
        
        Uses simple pattern matching on declarative sentences.
        """
        print(f"\n[Augmentor] Generating Q&A pairs...")
        traces = []
        
        # Entity patterns
        who_pattern = r'(\w+)\s+(went|said|saw|heard|did|made|took|gave)'
        what_pattern = r'(God|Jesus|Moses|David)\s+\w+\s+(\w+)'
        where_pattern = r'\s+(in|to|at|from)\s+([A-Z][a-z]+)'
        
        for verse in random.sample(self.verses, min(num_samples, len(self.verses))):
            text = verse['text']
            
            # Who question
            match = re.search(who_pattern, text)
            if match and random.random() < 0.3:
                entity = match.group(1)
                action = match.group(2)
                traces.append(ReasoningTrace(
                    input=f"Who {action} according to {verse['reference']}?",
                    output=entity,
                    reasoning_type="qa_who",
                    metadata={'source_verse': verse['reference']}
                ))
            
            # Where question
            match = re.search(where_pattern, text)
            if match and random.random() < 0.3:
                preposition = match.group(1)
                location = match.group(2)
                traces.append(ReasoningTrace(
                    input=f"Where is mentioned in {verse['reference']}?",
                    output=location,
                    reasoning_type="qa_where",
                    metadata={'source_verse': verse['reference']}
                ))
        
        print(f"  Generated {len(traces)} Q&A pairs")
        return traces
    
    def generate_cross_reference_chains(self, num_samples: int = 300) -> List[ReasoningTrace]:
        """
        Strategy 2: Cross-Reference Chains
        
        Link verses A → B → C to create transitive reasoning:
        "Verse A says X. Verse B says Y. Therefore, ..."
        
        Simulated cross-references (in real implementation, use Bible metadata).
        """
        print(f"\n[Augmentor] Generating cross-reference chains...")
        traces = []
        
        # Simulate cross-reference graph
        # In real implementation, parse actual cross-reference metadata
        for i in range(min(num_samples, len(self.verses) - 2)):
            verse_a = self.verses[i]
            verse_b = self.verses[i + 1]
            
            # Create implication chain
            traces.append(ReasoningTrace(
                input=f"{verse_a['reference']}: {verse_a['text'][:100]}... "
                      f"{verse_b['reference']}: {verse_b['text'][:100]}... "
                      f"What connection exists between these verses?",
                output=f"Both verses relate to the same theme discussed in {verse_a['reference']}.",
                reasoning_type="cross_reference",
                metadata={
                    'verse_a': verse_a['reference'],
                    'verse_b': verse_b['reference']
                }
            ))
        
        print(f"  Generated {len(traces)} cross-reference chains")
        return traces
    
    def generate_genealogy_reasoning(self, num_samples: int = 200) -> List[ReasoningTrace]:
        """
        Strategy 4: Genealogical Graph Traversal
        
        Extract genealogical relationships and create reasoning tasks:
        "Abraham fathered Isaac. Isaac fathered Jacob. Who is Jacob's grandfather?"
        """
        print(f"\n[Augmentor] Generating genealogy reasoning...")
        traces = []
        
        # Extract "X fathered/begat Y" patterns
        genealogy_pattern = r'(\w+)\s+(?:fathered|begat|was the father of)\s+(\w+)'
        
        relationships = []
        for verse in self.verses:
            matches = re.finditer(genealogy_pattern, verse['text'], re.IGNORECASE)
            for match in matches:
                parent = match.group(1)
                child = match.group(2)
                relationships.append((parent, child, verse['reference']))
        
        # Build transitivity chains
        parent_map = defaultdict(list)
        for parent, child, ref in relationships:
            parent_map[parent].append((child, ref))
        
        # Generate transitive questions
        for parent, children in list(parent_map.items())[:num_samples]:
            for child, ref in children:
                # Check if child is also a parent
                if child in parent_map:
                    grandchildren = parent_map[child]
                    for grandchild, ref2 in grandchildren:
                        traces.append(ReasoningTrace(
                            input=f"{parent} fathered {child}. {child} fathered {grandchild}. "
                                  f"Who is {grandchild}'s grandfather?",
                            output=parent,
                            reasoning_type="genealogy_transitive",
                            metadata={
                                'grandfather': parent,
                                'father': child,
                                'grandson': grandchild
                            }
                        ))
                        
                        if len(traces) >= num_samples:
                            break
                
                if len(traces) >= num_samples:
                    break
            
            if len(traces) >= num_samples:
                break
        
        print(f"  Generated {len(traces)} genealogy traces")
        return traces
    
    def generate_adversarial_samples(self, num_samples: int = 100) -> List[ReasoningTrace]:
        """
        Strategy 3: Adversarial Contradictions
        
        Generate theologically inconsistent statements for contrastive learning:
        "The soul is immortal." (Incorrect per NWT)
        "The soul is mortal." (Correct per NWT)
        """
        print(f"\n[Augmentor] Generating adversarial samples...")
        traces = []
        
        # NWT-specific doctrinal assertions (simplified)
        correct_incorrect_pairs = [
            (
                "The soul that is sinning will die.",
                "The soul that is sinning will suffer forever.",
                "soul_doctrine"
            ),
            (
                "Jehovah is the one true God.",
                "There are multiple gods equal to Jehovah.",
                "monotheism"
            ),
            (
                "Jesus is the Son of God.",
                "Jesus is God Himself.",
                "trinity_rejection"
            ),
        ]
        
        for correct, incorrect, category in correct_incorrect_pairs:
            # Positive example
            traces.append(ReasoningTrace(
                input=f"Is the following statement consistent with NWT theology? {correct}",
                output="Yes, this aligns with NWT doctrine.",
                reasoning_type="adversarial_positive",
                metadata={'category': category}
            ))
            
            # Negative example
            traces.append(ReasoningTrace(
                input=f"Is the following statement consistent with NWT theology? {incorrect}",
                output="No, this contradicts NWT doctrine.",
                reasoning_type="adversarial_negative",
                metadata={'category': category}
            ))
        
        print(f"  Generated {len(traces)} adversarial samples")
        return traces
    
    def generate_all(self) -> List[ReasoningTrace]:
        """Generate all augmentation strategies."""
        print(f"\n{'='*60}")
        print(f"Arbiter Data Augmentation")
        print(f"{'='*60}")
        print(f"Corpus: {self.corpus_path}")
        print(f"Base verses: {len(self.verses)}")
        
        all_traces = []
        
        all_traces.extend(self.generate_qa_pairs(500))
        all_traces.extend(self.generate_cross_reference_chains(300))
        all_traces.extend(self.generate_genealogy_reasoning(200))
        all_traces.extend(self.generate_adversarial_samples(100))
        
        print(f"\n{'='*60}")
        print(f"Total traces generated: {len(all_traces)}")
        print(f"{'='*60}\n")
        
        return all_traces
    
    def save_jsonl(self, traces: List[ReasoningTrace], output_path: str):
        """Save traces to JSONL format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + '\n')
        
        print(f"[DataAugmentor] Saved {len(traces)} traces to {output_path}")
    
    def save_human_readable(self, traces: List[ReasoningTrace], output_path: str):
        """Save traces in human-readable format for inspection."""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, trace in enumerate(traces[:50], 1):  # First 50 for inspection
                f.write(f"\n{'='*60}\n")
                f.write(f"Trace {i}: {trace.reasoning_type}\n")
                f.write(f"{'='*60}\n")
                f.write(f"INPUT:\n{trace.input}\n\n")
                f.write(f"OUTPUT:\n{trace.output}\n")
                if trace.metadata:
                    f.write(f"\nMETADATA: {trace.metadata}\n")
        
        print(f"[DataAugmentor] Saved sample to {output_path}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arbiter Data Augmentation")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus text file")
    parser.add_argument("--output", type=str, default="./augmented_data.jsonl",
                       help="Output JSONL path")
    parser.add_argument("--human-readable", type=str, default="./augmented_sample.txt",
                       help="Human-readable sample output")
    
    args = parser.parse_args()
    
    # Initialize augmentor
    augmentor = ArbiterDataAugmentor(corpus_path=args.corpus)
    
    # Generate all traces
    traces = augmentor.generate_all()
    
    # Save outputs
    augmentor.save_jsonl(traces, args.output)
    augmentor.save_human_readable(traces, args.human_readable)
    
    print(f"\n✓ Data augmentation complete")
    print(f"  JSONL: {args.output}")
    print(f"  Sample: {args.human_readable}")
