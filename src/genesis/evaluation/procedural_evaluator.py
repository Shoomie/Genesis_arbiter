import torch
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import math

class ProceduralEvaluator:
    """
    Procedural evaluator for Genesis Arbiter.
    Runs masking-based reconstruction and structural probes using the full corpus.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        tokenizer,
        device: torch.device,
        verbose: bool = True
    ):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        
        # Determine mask token
        if (hasattr(tokenizer, 'is_character_level') and tokenizer.is_character_level()) or \
           (hasattr(tokenizer, 'is_byte_level') and tokenizer.is_byte_level()):
            self.mask_token_id = getattr(tokenizer, 'mask_id', 0)
        else:
            try:
                self.mask_token_id = tokenizer.tokenizer.token_to_id("[MASK]")
            except:
                self.mask_token_id = 0
            
    def _calculate_cer(self, pred_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Calculate Character Error Rate (CER) at token/character level."""
        # Simple CER: (errors / total)
        # In character-level models, this is literally CER.
        # In BPE, it's Token Error Rate, but we'll call it CER for consistency if requested.
        errors = (pred_ids != target_ids).sum().item()
        total = target_ids.numel()
        return errors / total if total > 0 else 0.0

    def _mask_sequence(
        self, 
        tokens: torch.Tensor, 
        strategy: str = 'bert',
        mask_prob: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking strategy to a sequence."""
        masked_tokens = tokens.clone()
        labels = torch.full_like(tokens, -100)
        
        seq_len = tokens.size(0)
        
        if strategy == 'bert':
            # 15% random tokens masked
            mask = torch.rand(seq_len, device=tokens.device) < mask_prob
            labels[mask] = tokens[mask]
            masked_tokens[mask] = self.mask_token_id
            
        elif strategy == 'span':
            # Mask consecutive spans (approx 15-20% total)
            num_to_mask = int(seq_len * mask_prob)
            span_len = 5 # Average span length
            num_spans = max(1, num_to_mask // span_len)
            
            for _ in range(num_spans):
                start = torch.randint(0, max(1, seq_len - span_len), (1,)).item()
                actual_span = min(span_len, seq_len - start)
                labels[start:start+actual_span] = tokens[start:start+actual_span]
                masked_tokens[start:start+actual_span] = self.mask_token_id
                
        elif strategy == 'verse':
            # Mask entire verse if multi-verse context exists
            # For evaluation on single verses, this just masks everything.
            # If the eval batch contains a sequence of verses, we mask 1.
            labels[:] = tokens[:]
            masked_tokens[:] = self.mask_token_id
            
        return masked_tokens, labels

    def evaluate_reconstruction(
        self, 
        num_samples: int = 50, 
        strategy: str = 'bert',
        context_size: int = 3,
        use_amp: bool = True,
        amp_dtype = torch.float16
    ) -> Dict[str, float]:
        """Run reconstruction evaluation."""
        self.model.eval()
        total_cer = 0.0
        total_exact = 0.0
        total_loss = 0.0
        count = 0
        
        # Use a fixed seed for reproducibility across steps
        rng = np.random.RandomState(42)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 1. Sample primary verse
                idx = rng.randint(0, len(self.dataset.verse_indices))
                
                # 2. Add context if requested
                if strategy == 'verse' or context_size > 1:
                    # Try to get consecutive verses
                    start_idx = max(0, idx - context_size // 2)
                    end_idx = min(len(self.dataset.verse_indices), start_idx + context_size)
                    
                    combined_tokens = []
                    target_verse_start_in_seq = 0
                    target_verse_len = 0
                    
                    for i in range(start_idx, end_idx):
                        s, l = self.dataset.verse_indices[i]
                        tokens = self.dataset.data_tensor[s : s + l]
                        if i == idx:
                            target_verse_start_in_seq = sum(len(t) for t in combined_tokens)
                            target_verse_len = len(tokens)
                        combined_tokens.append(tokens)
                    
                    raw_tokens = torch.cat(combined_tokens).to(self.device)
                    
                    # Truncate to model max if too long
                    if len(raw_tokens) > 512:
                        raw_tokens = raw_tokens[:512]
                        # Adjust target if needed (simplified: just skip if target is truncated)
                        if target_verse_start_in_seq + target_verse_len > 512:
                            continue
                else:
                    start, length = self.dataset.verse_indices[idx]
                    raw_tokens = self.dataset.data_tensor[start : start + length].to(self.device)
                    target_verse_start_in_seq = 0
                    target_verse_len = length
                
                if len(raw_tokens) < 5:
                    if self.verbose: print(f"    [DEBUG] Sample too short: {len(raw_tokens)}")
                    continue 
                
                # 3. Apply masking
                if strategy == 'verse':
                    # Mask exactly the target verse within context
                    masked = raw_tokens.clone()
                    labels = torch.full_like(raw_tokens, -100)
                    labels[target_verse_start_in_seq : target_verse_start_in_seq + target_verse_len] = \
                        raw_tokens[target_verse_start_in_seq : target_verse_start_in_seq + target_verse_len]
                    masked[target_verse_start_in_seq : target_verse_start_in_seq + target_verse_len] = self.mask_token_id
                else:
                    masked, labels = self._mask_sequence(raw_tokens, strategy=strategy)
                
                # 4. Forward pass
                input_ids = masked.unsqueeze(0)
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    loss, logits = self.model.forward_lm(input_ids, labels.unsqueeze(0))
                    
                # 5. Calculate metrics
                mask = labels != -100
                if not mask.any():
                    if self.verbose: print(f"    [DEBUG] No tokens masked for strategy {strategy}")
                    continue
                
                preds = logits[0, mask].argmax(dim=-1)
                targets = labels[mask]
                
                total_loss += loss.item()
                total_cer += self._calculate_cer(preds, targets)
                total_exact += 1.0 if torch.equal(preds, targets) else 0.0
                count += 1
                
        if self.verbose:
            print(f"    [DEBUG] Strategy {strategy} finished with count={count}")
                
        prefix = f'recon_{strategy}'
        return {
            f'{prefix}_loss': total_loss / count if count > 0 else 0.0,
            f'{prefix}_cer': total_cer / count if count > 0 else 0.0,
            f'{prefix}_exact': total_exact / count if count > 0 else 0.0
        }

    def evaluate_auxiliary_tasks(
        self,
        num_samples: int = 20,
        use_amp: bool = True,
        amp_dtype = torch.float16
    ) -> Dict[str, float]:
        """Run procedural versions of auxiliary tasks."""
        self.model.eval()
        metrics = {}
        
        # 1. Coherence
        coherent_acc = 0.0
        coherence_loss = 0.0
        for _ in range(num_samples):
            batch = self.dataset._get_coherence_sample()
            with torch.no_grad():
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    loss, logits = self.model.forward_coherence(
                        batch['verse1_tokens'], batch['verse2_tokens'], batch['labels']
                    )
            pred = (torch.sigmoid(logits) > 0.5).float()
            coherent_acc += (pred == batch['labels']).float().mean().item()
            coherence_loss += loss.item()
        metrics['val_coherence_acc'] = coherent_acc / num_samples
        metrics['val_coherence_loss'] = coherence_loss / num_samples
        
        # 2. Cross-Reference (Triplet)
        cross_ref_loss = 0.0
        for _ in range(num_samples):
            batch = self.dataset._get_cross_ref_sample()
            with torch.no_grad():
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                   loss, _ = self.model.forward_cross_ref(
                       batch['anchor_tokens'], batch['positive_tokens'], batch['negative_tokens']
                   )
            cross_ref_loss += loss.item()
        metrics['val_cross_ref_loss'] = cross_ref_loss / num_samples
        
        # 3. Paraphrase (if available)
        if len(self.dataset.locale_verse_map) >= 2:
            para_acc = 0.0
            para_loss = 0.0
            for _ in range(num_samples):
                batch = self.dataset._get_paraphrase_sample()
                with torch.no_grad():
                    with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        loss, sim = self.model.forward_paraphrase(
                            batch['verse1_tokens'], batch['verse2_tokens'], batch['labels']
                        )
                pred = (sim > 0.5).float()
                para_acc += (pred == batch['labels']).float().mean().item()
                para_loss += loss.item()
            metrics['val_paraphrase_acc'] = para_acc / num_samples
            metrics['val_paraphrase_loss'] = para_loss / num_samples
            
        return metrics

    def run_suite(
        self, 
        use_amp: bool = True, 
        amp_dtype = torch.float16,
        num_recon_samples: int = 50,
        num_aux_samples: int = 20
    ) -> Dict[str, float]:
        """Run full evaluation suite."""
        if self.verbose:
            print(f"  [EVAL] Running procedural reconstruction (BERT, n={num_recon_samples})...")
        bert_metrics = self.evaluate_reconstruction(strategy='bert', num_samples=num_recon_samples, use_amp=use_amp, amp_dtype=amp_dtype)
        
        if self.verbose:
            print(f"  [EVAL] Running procedural reconstruction (Span, n={num_recon_samples})...")
        span_metrics = self.evaluate_reconstruction(strategy='span', num_samples=num_recon_samples, use_amp=use_amp, amp_dtype=amp_dtype)
        
        if self.verbose:
            print(f"  [EVAL] Running procedural reconstruction (Verse, n={num_recon_samples})...")
        verse_metrics = self.evaluate_reconstruction(strategy='verse', num_samples=num_recon_samples, context_size=3, use_amp=use_amp, amp_dtype=amp_dtype)
        
        if self.verbose:
            print(f"  [EVAL] Running structural probes (n={num_aux_samples})...")
        aux_metrics = self.evaluate_auxiliary_tasks(num_samples=num_aux_samples, use_amp=use_amp, amp_dtype=amp_dtype)
        
        # Combine all
        all_metrics = {**bert_metrics, **span_metrics, **verse_metrics, **aux_metrics}
        
        # Summary log line
        if self.verbose:
            print(f"  [EVAL] BERT-Acc: {all_metrics['recon_bert_exact']:.4f} | "
                  f"Verse-Acc: {all_metrics['recon_verse_exact']:.4f} | "
                  f"CER: {all_metrics['recon_bert_cer']:.4f}")
                  
        return all_metrics
