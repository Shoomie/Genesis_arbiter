#!/usr/bin/env python3
"""
Genesis Central Menu
====================
A unified interface for all Genesis project functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

# ANSI color codes for terminal styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the Genesis project header."""
    # Set UTF-8 encoding for Windows console
    if os.name == 'nt':
        try:
            os.system('chcp 65001 >nul 2>&1')
        except:
            pass
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 70)
    print()
    print("   ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗")
    print("  ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝")
    print("  ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗")
    print("  ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║")
    print("  ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║")
    print("   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝")
    print()
    print("=" * 70)
    print(f"{Colors.ENDC}")
    print(f"{Colors.HEADER}       Bible-Trained Language Models for Causal Reasoning{Colors.ENDC}")
    print()

def print_menu():
    """Print the main menu options."""
    print(f"{Colors.BOLD}Main Menu:{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}[1]{Colors.ENDC} {Colors.BOLD}Train Model{Colors.ENDC}")
    print("    ├─ [1a] FlashAttention Training (3-4x faster)")
    print("    └─ [1b] Multi-Task Training (with grokking detection)\n")
    
    print(f"{Colors.GREEN}[2]{Colors.ENDC} {Colors.BOLD}Corpus Analysis{Colors.ENDC}")
    print("    ├─ [2a] Count unique words")
    print("    └─ [2b] Count logical connectives\n")
    
    print(f"{Colors.GREEN}[3]{Colors.ENDC} {Colors.BOLD}Evaluation & Testing{Colors.ENDC}")
    print("    ├─ [3a] Calculate perplexity")
    print("    └─ [3b] Run friction stress test\n")
    
    print(f"{Colors.GREEN}[4]{Colors.ENDC} {Colors.BOLD}Documentation{Colors.ENDC}")
    print("    ├─ [4a] Quick Reference Guide")
    print("    ├─ [4b] Theoretical Foundations")
    print("    ├─ [4c] Setup Guide (FlashAttention)")
    print("    ├─ [4d] Grokking Detection Methodology")
    print("    └─ [4e] Research Roadmap\n")
    
    print(f"{Colors.YELLOW}[5]{Colors.ENDC} {Colors.BOLD}Arbiter Automation{Colors.ENDC}")
    print("    ├─ [5a] Verify FlashAttention Setup")
    print("    ├─ [5b] Quick Evaluation")
    print("    ├─ [5c] Long Training Pipeline")
    print("    ├─ [5d] Parameter Sweep")
    print("    └─ [5e] Data Augmentation\n")
    
    print(f"{Colors.YELLOW}[6]{Colors.ENDC} {Colors.BOLD}Project Information{Colors.ENDC}")
    print("    View project statistics and status\n")
    
    print(f"{Colors.RED}[0]{Colors.ENDC} {Colors.BOLD}Exit{Colors.ENDC}\n")
    print("=" * 70)

def run_script(script_path, description):
    """Run a Python script with proper error handling."""
    print(f"\n{Colors.CYAN}▶ {description}...{Colors.ENDC}\n")
    try:
        # Change to script directory for relative imports
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        if script_dir:
            subprocess.run([sys.executable, script_name], cwd=script_dir, check=True)
        else:
            subprocess.run([sys.executable, script_path], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}✗ Error: Script exited with code {e.returncode}{Colors.ENDC}")
    except FileNotFoundError:
        print(f"\n{Colors.RED}✗ Error: Script not found: {script_path}{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

def open_documentation(doc_path):
    """Open a documentation file in the default viewer."""
    if not os.path.exists(doc_path):
        print(f"\n{Colors.RED}✗ Error: File not found: {doc_path}{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
        return
    
    print(f"\n{Colors.CYAN}▶ Opening {os.path.basename(doc_path)}...{Colors.ENDC}")
    
    try:
        if sys.platform == 'win32':
            os.startfile(doc_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', doc_path])
        else:  # Linux
            subprocess.run(['xdg-open', doc_path])
        
        print(f"{Colors.GREEN}✓ Document opened in default application{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not open automatically. Path: {doc_path}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

def show_project_info():
    """Display project statistics and information."""
    clear_screen()
    print_header()
    
    print(f"{Colors.BOLD}📊 Project Statistics{Colors.ENDC}\n")
    
    # Count files and directories
    engine_files = sum(1 for _ in Path('engine').rglob('*.py'))
    doc_files = sum(1 for _ in Path('docs').rglob('*.md'))
    script_files = sum(1 for _ in Path('scripts').rglob('*.py'))
    
    print(f"  {Colors.CYAN}Engine Files:{Colors.ENDC} {engine_files} Python modules")
    print(f"  {Colors.CYAN}Documentation:{Colors.ENDC} {doc_files} markdown files")
    print(f"  {Colors.CYAN}Utility Scripts:{Colors.ENDC} {script_files} analysis tools")
    
    # Corpus information
    corpus_path = Path('engine/nwt_corpus.txt')
    if corpus_path.exists():
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print(f"  {Colors.CYAN}Corpus Size:{Colors.ENDC} {size_mb:.2f} MB")
    
    # Check for checkpoints
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoint_count = sum(1 for _ in checkpoint_dir.rglob('*.pt'))
        print(f"  {Colors.CYAN}Saved Checkpoints:{Colors.ENDC} {checkpoint_count}")
    
    print(f"\n{Colors.BOLD}📁 Project Structure:{Colors.ENDC}\n")
    print("  Genesis_arbiter/")
    print("  ├── engine/              # Core training system")
    print("  │   ├── models/          # Transformer architectures")
    print("  │   ├── training/        # FlashAttention & callbacks")
    print("  │   ├── datasets/        # Data loading")
    print("  │   └── train_*.py       # Training scripts")
    print("  ├── docs/")
    print("  │   ├── research/        # Technical papers")
    print("  │   └── reference/       # Quick guides")
    print("  ├── scripts/             # Utility tools")
    print("  └── checkpoints/         # Model snapshots")
    
    print(f"\n{Colors.BOLD}🎯 Framework Status:{Colors.ENDC}")
    print(f"  {Colors.GREEN}✅ Phase 1:{Colors.ENDC} FlashAttention Integration (3-4x speedup)")
    print(f"  {Colors.GREEN}✅ Phase 2:{Colors.ENDC} Multi-Task Learning (142 translations)")
    print(f"  {Colors.GREEN}✅ Phase 3:{Colors.ENDC} Grokking Detection & Monitoring")
    
    print(f"\n{Colors.BOLD}📌 Ready for:{Colors.ENDC} Production training runs with automated grokking detection")
    
    input(f"\n{Colors.YELLOW}Press Enter to return to menu...{Colors.ENDC}")

def main():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input(f"{Colors.BOLD}Select an option: {Colors.ENDC}").strip().lower()
        
        if choice == '1a':
            run_script('engine/train_composer.py', 'Launching FlashAttention Training System')
        
        elif choice == '1b':
            run_script('engine/train_multi_task.py', 'Launching Multi-Task Training with Grokking Detection')
        
        elif choice == '2a':
            run_script('scripts/count_unique_words.py', 'Analyzing Corpus Vocabulary')
        
        elif choice == '2b':
            run_script('scripts/count_logical_connectives.py', 'Counting Logical Connectives')
        
        elif choice == '3a':
            run_script('scripts/arbiter_perplexity.py', 'Calculating Model Perplexity')
        
        elif choice == '3b':
            run_script('scripts/friction_stress_test.py', 'Running Friction Stress Test')
        
        elif choice == '4a':
            open_documentation('docs/reference/QUICK_REFERENCE.md')
        
        elif choice == '4b':
            open_documentation('docs/research/theoretical_foundations.md')
        
        elif choice == '4c':
            open_documentation('PHASE1_SETUP.md')
        
        elif choice == '4d':
            open_documentation('docs/research/grokking_detection_methodology.md')
        
        elif choice == '4e':
            open_documentation('docs/roadmap/README.md')
        
        elif choice == '5a':
            run_script('engine/verify_phase1.py', 'Verifying FlashAttention Setup')
        
        elif choice == '5b':
            run_script('engine/arbiter_quick_eval.py', 'Running Quick Evaluation')
        
        elif choice == '5c':
            run_script('engine/arbiter_long_pipeline.py', 'Launching Long Training Pipeline')
        
        elif choice == '5d':
            run_script('engine/arbiter_sweep_orchestrator.py', 'Starting Parameter Sweep')
        
        elif choice == '5e':
            run_script('scripts/arbiter_data_augmentor.py', 'Generating Augmented Data')
        
        elif choice == '6':
            show_project_info()
        
        elif choice == '0':
            print(f"\n{Colors.CYAN}Thank you for using Genesis!{Colors.ENDC}")
            print(f"{Colors.YELLOW}Deep Reasoning in Data-Constrained Regimes{Colors.ENDC}\n")
            sys.exit(0)
        
        else:
            print(f"\n{Colors.RED}Invalid option. Please try again.{Colors.ENDC}")
            input(f"{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Exiting...{Colors.ENDC}\n")
        sys.exit(0)

