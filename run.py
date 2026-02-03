#!/usr/bin/env python3
"""
Genesis Central Menu
====================
A unified interface for all Genesis project functionality.
"""

import os
import sys
import subprocess
import toml
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

def load_genesis_config():
    """Load central configuration from genesis_config.toml."""
    config_path = Path(__file__).parent / "genesis_config.toml"
    if not config_path.exists():
        return {}
    try:
        return toml.load(config_path)
    except Exception as e:
        print(f"\033[91m[!] Error loading genesis_config.toml: {e}\033[0m")
        return {}

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
    print("                     G E N E S I S")
    print("              Bible-Trained Language Models")
    print()
    print("=" * 70)
    print(f"{Colors.ENDC}")
    print(f"{Colors.HEADER}       Bible-Trained Language Models for Causal Reasoning{Colors.ENDC}")
    print()

def print_menu():
    """Print the main menu options."""
    print(f"{Colors.BOLD}Main Menu:{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}[1]{Colors.ENDC} {Colors.BOLD}Train Model{Colors.ENDC}")
    print("    - [1a] Standard training (Fast)")
    print("    - [1b] Full evaluation training\n")
    
    print(f"{Colors.GREEN}[2]{Colors.ENDC} {Colors.BOLD}Evaluation & Testing{Colors.ENDC}")
    print("    - [2a] Interactive Checkpoint Chat")
    print("    - [2b] Run friction stress test")
    print("    - [2c] Calculate perplexity\n")
    
    print(f"{Colors.GREEN}[3]{Colors.ENDC} {Colors.BOLD}Corpus Analysis{Colors.ENDC}")
    print("    - [3a] Count unique words")
    print("    - [3b] Count logical connectives")
    print("    - [3c] Pre-process Data (Create Training Data Cache File)")
    print("    - [3d] Generate WWM Boundary Map (Level-Up Training)\n")
    
    print(f"{Colors.GREEN}[4]{Colors.ENDC} {Colors.BOLD}Documentation{Colors.ENDC}")
    print("    - [4a] Quick Reference Guide")
    print("    - [4b] Theoretical Foundations")
    print("    - [4c] Setup Guide (FlashAttention)")
    print("    - [4d] Grokking Detection Methodology")
    print("    - [4e] Research Roadmap\n")
    
    print(f"{Colors.YELLOW}[5]{Colors.ENDC} {Colors.BOLD}Arbiter Automation (under construction){Colors.ENDC}")
    print("    - [5a] Verify FlashAttention Setup")
    print("    - [5b] Quick Evaluation")
    print("    - [5c] Long Training Pipeline")
    print("    - [5d] Parameter Sweep")
    print("    - [5e] Data Augmentation\n")
    
    print(f"{Colors.YELLOW}[6]{Colors.ENDC} {Colors.BOLD}Project Information{Colors.ENDC}")
    print("    View project statistics and status\n")
    
    print(f"{Colors.RED}[0]{Colors.ENDC} {Colors.BOLD}Exit{Colors.ENDC}\n")
    print("=" * 70)

def run_script(script_path, description, args=None):
    """Run a Python script with proper error handling."""
    print(f"\n{Colors.CYAN}> {description}...{Colors.ENDC}\n")
    if args is None:
        args = []
        
    try:
        # Change to script directory for relative imports
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        # Add 'src' and root to PYTHONPATH so 'from genesis.xxx' and 'from tools.xxx' work
        env = os.environ.copy()
        project_root_str = str(Path(__file__).parent.absolute())
        src_path = str(Path(__file__).parent.absolute() / "src")
        
        python_path = f"{src_path}{os.pathsep}{project_root_str}"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = python_path

        # Prepare command
        if script_dir:
            cmd = [sys.executable, script_name] + args
            subprocess.run(cmd, cwd=script_dir, check=True, env=env)
        else:
            cmd = [sys.executable, script_path] + args
            subprocess.run(cmd, check=True, env=env)
            
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}[X] Error: Script exited with code {e.returncode}{Colors.ENDC}")
    except FileNotFoundError:
        print(f"\n{Colors.RED}[X] Error: Script not found: {script_path}{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}[X] Error: {str(e)}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

def open_documentation(doc_path):
    """Open a documentation file in the default viewer."""
    if not os.path.exists(doc_path):
        print(f"\n{Colors.RED}[X] Error: File not found: {doc_path}{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
        return
    
    print(f"\n{Colors.CYAN}> Opening {os.path.basename(doc_path)}...{Colors.ENDC}")
    
    try:
        if sys.platform == 'win32':
            os.startfile(doc_path)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', doc_path])
        else:  # Linux
            subprocess.run(['xdg-open', doc_path])
        
        print(f"{Colors.GREEN}[OK] Document opened in default application{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.YELLOW}[!] Could not open automatically. Path: {doc_path}{Colors.ENDC}")
    
    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

def show_project_info():
    """Display project statistics and information."""
    clear_screen()
    print_header()
    
    print(f"{Colors.BOLD}[Stats] Project Statistics{Colors.ENDC}\n")
    
    # Count files and directories
    engine_files = sum(1 for _ in Path('src/genesis').rglob('*.py'))
    doc_files = sum(1 for _ in Path('docs').rglob('*.md')) + sum(1 for _ in Path('project_doc').rglob('*.md'))
    script_files = sum(1 for _ in Path('tools').rglob('*.py'))
    
    print(f"  {Colors.CYAN}Core Package (genesis):{Colors.ENDC} {engine_files} Python modules")
    print(f"  {Colors.CYAN}Documentation:{Colors.ENDC} {doc_files} research & meta files")
    print(f"  {Colors.CYAN}Utility Tools:{Colors.ENDC} {script_files} analysis tools")
    
    # Corpus information
    corpus_path = Path('data/genesis_data_cache.pt')
    if corpus_path.exists():
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print(f"  {Colors.CYAN}Data Cache Size:{Colors.ENDC} {size_mb:.2f} MB")
    
    # Check for checkpoints
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoint_count = sum(1 for _ in checkpoint_dir.rglob('*.pt'))
        print(f"  {Colors.CYAN}Saved Checkpoints:{Colors.ENDC} {checkpoint_count}")
    
    print(f"\n{Colors.BOLD}[Folders] Project Structure:{Colors.ENDC}\n")
    print("  Genesis_arbiter/")
    print("  ├── src/genesis/         # Core genesis package")
    print("  │   ├── models/          # Transformer architectures")
    print("  │   ├── training/        # FlashAttention & callbacks")
    print("  │   ├── datasets/        # Data loading")
    print("  │   └── train_*.py       # Training scripts")
    print("  ├── data/                # Tokenizers and data caches")
    print("  ├── tools/               # Utility tools")
    print("  ├── docs/                # Research papers & guides")
    print("  ├── project_doc/         # Legal & Contribution info")
    print("  └── checkpoints/         # Model snapshots")
    
    print(f"\n{Colors.BOLD}[Status] Framework Status:{Colors.ENDC}")
    print(f"  {Colors.GREEN}[OK] Phase 1:{Colors.ENDC} FlashAttention Integration (3-4x speedup)")
    print(f"  {Colors.GREEN}[OK] Phase 2:{Colors.ENDC} Multi-Task Learning (142 translations)")
    print(f"  {Colors.GREEN}[OK] Phase 3:{Colors.ENDC} Grokking Detection & Monitoring")
    
    print(f"\n{Colors.BOLD}[Ready] Ready for:{Colors.ENDC} Production training runs with automated grokking detection")
    
    input(f"\n{Colors.YELLOW}Press Enter to return to menu...{Colors.ENDC}")

def main():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input(f"{Colors.BOLD}Select an option: {Colors.ENDC}").strip().lower()
        
        if choice in ['1a', '1b']:
            config = load_genesis_config()
            train_cfg = config.get("training", {})
            sys_cfg = config.get("system", {})
            eval_cfg = config.get("evaluation", {})
            
            # Map config to command line arguments
            args = []
            if train_cfg.get("resume", True): args.append("--resume")
            if "mode" in train_cfg: args.extend(["--mode", train_cfg["mode"]])
            if "batch_size" in train_cfg: args.extend(["--batch-size", str(train_cfg["batch_size"])])
            if "learning_rate" in train_cfg: args.extend(["--lr", str(train_cfg["learning_rate"])])
            if "weight_decay" in train_cfg: args.extend(["--weight-decay", str(train_cfg["weight_decay"])])
            if "max_steps" in train_cfg: args.extend(["--steps", str(train_cfg["max_steps"])])
            if "val_interval" in eval_cfg: args.extend(["--val-interval", str(eval_cfg["val_interval"])])
            
            if choice == '1a':
                # Disable extensive evaluation for 1a
                args.extend(["--eval-interval", "0"])
                desc = 'Standard training (Fast)'
            else:
                if "eval_interval" in eval_cfg: args.extend(["--eval-interval", str(eval_cfg["eval_interval"])])
                desc = 'Full evaluation training'

            if sys_cfg.get("compile_model"): args.append("--compile")
            if sys_cfg.get("gradient_checkpointing"): args.append("--gradient-checkpointing")
            
            print(f"\n{Colors.CYAN}Starting {desc} (Global Config: genesis_config.toml)...{Colors.ENDC}")
            run_script('src/genesis/train.py', desc, args=args)
        
        elif choice == '2a':
            config = load_genesis_config()
            interact_cfg = config.get("inference", {})
            
            args = []
            if "device" in interact_cfg: args.extend(["--device", interact_cfg["device"]])
            if "temperature" in interact_cfg: args.extend(["--temp", str(interact_cfg["temperature"])])
            if "max_tokens" in interact_cfg: args.extend(["--max-tokens", str(interact_cfg["max_tokens"])])
            
            run_script('tools/interact_with_checkpoint.py', 'Interactive Checkpoint Chat', args=args)
            
        elif choice == '2b':
            run_script('tools/friction_stress_test.py', 'Running Axiomatic Friction Stress Test')
            
        elif choice == '2c':
            run_script('tools/arbiter_perplexity.py', 'Calculating Perplexity on Contrastive Datasets')
            
        elif choice == '3a':
            run_script('tools/count_unique_words.py', 'Analyzing Corpus Vocabulary')
            
        elif choice == '3b':
            run_script('tools/count_logical_connectives.py', 'Analyzing Logical Connectives in Dataset')
            
        elif choice == '3c':
            run_script('tools/update_data_cache.py', 'Updating VRAM Data Cache')
        
        elif choice == '3d':
            run_script('tools/generate_boundary_map.py', 'Generating Whole-Word Masking (WWM) Boundary Map')
        
        elif choice == '4a':
            open_documentation('docs/reference/QUICK_REFERENCE.md')
        
        elif choice == '4b':
            open_documentation('docs/research/theoretical_foundations.md')
        
        elif choice == '4c':
            open_documentation('docs/PHASE1_SETUP.md')
        
        elif choice == '4d':
            open_documentation('docs/research/grokking_detection_methodology.md')
        
        elif choice == '4e':
            open_documentation('docs/roadmap/README.md')
        
        elif choice == '5a':
            run_script('src/genesis/verify.py', 'Verifying FlashAttention Setup')
        
        elif choice == '5b':
            run_script('src/genesis/pipelines/quick_eval.py', 'Running Quick Evaluation')
        
        elif choice == '5c':
            run_script('src/genesis/pipelines/long_pipeline.py', 'Launching Long Training Pipeline')
        
        elif choice == '5d':
            run_script('src/genesis/pipelines/sweep.py', 'Starting Parameter Sweep')
        
        elif choice == '5e':
            run_script('tools/arbiter_data_augmentor.py', 'Generating Augmented Data')
        
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

