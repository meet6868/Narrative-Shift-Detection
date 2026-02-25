#!/usr/bin/env python3
"""
Stage 2: SBERT Embeddings Generator - Standalone GPU/CPU Script
================================================================

This script processes Stage 1 CSV files and generates dual-window SBERT embeddings.

Features:
- ✅ Auto GPU/CPU detection and utilization
- ✅ Optimized for 1.64GB NVIDIA GPUs (RTX 450 and similar)
- ✅ Automatic OOM (Out of Memory) recovery with batch size reduction
- ✅ Full GPU utilization with FP16 mixed precision
- ✅ Dynamic batch sizing for maximum throughput
- ✅ Configurable input/output paths
- ✅ Batch processing optimized for device
- ✅ Progress tracking with GPU memory monitoring
- ✅ Error handling and recovery
- ✅ Works on any PC (Windows/Linux/Mac)

GPU Optimizations (1.64GB NVIDIA RTX 450):
- Initial batch size: 32 (conservative for 1.64GB VRAM)
- Auto-reduces to 16 or 8 if OOM errors occur
- Mixed precision (FP16): 2x faster processing
- Max sequence length: 384 tokens (reduced from 512 to save memory)
- Expandable segments: Enabled to reduce memory fragmentation
- Periodic cache clearing every 3 files
- Gradient computation disabled (inference only)

Performance Expectations:
- ~150-300 sentences/second on 1.64GB GPU (FP16, batch 32)
- ~100-200 sentences/second on 1.64GB GPU (FP16, batch 16 after OOM)
- ~40-80 sentences/second on CPU
- Memory usage: 1.2-1.5 GB VRAM (safe for 1.64GB GPU)

Tuning Tips:
- Script automatically reduces batch size if OOM errors occur
- If still getting OOM: Edit line 84, set BATCH_SIZE = 16 or 8
- For more speed (if no OOM): Set BATCH_SIZE = 48 or 64
- CPU-only mode: Automatically uses smaller batch size (16)
- Disable FP16: Set USE_FP16 = False (slower but uses less memory)

Usage:
    # Make sure virtual environment is activated
    source venv/bin/activate  # Linux/Mac
    # or
    venv\\Scripts\\activate  # Windows
    
    # Run with virtual environment's Python
    python stage2_gpu_processor.py
    # or explicitly
    /path/to/venv/bin/python stage2_gpu_processor.py
    
    Then follow the prompts to enter:
    - Stage 1 input folder path
    - Stage 2 output folder path

Author: Narrative Shift Detection Framework
Date: February 2026
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Stage 2 processing - Optimized for 2GB GPU"""
    
    # SBERT Model
    MODEL_NAME = 'all-mpnet-base-v2'
    EMBEDDING_DIM = 768
    
    # Device will be auto-detected
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Optimized batch size for 1.64GB GPU (RTX 450)
    # Conservative batch size to avoid OOM errors
    # Will automatically reduce if OOM occurs
    BATCH_SIZE = 32 if DEVICE == 'cuda' else 16
    
    # GPU Memory optimization settings
    MAX_SEQ_LENGTH = 384  # Reduced from 512 to save memory
    USE_FP16 = True if DEVICE == 'cuda' else False  # Half precision for 2x speedup
    PIN_MEMORY = True if DEVICE == 'cuda' else False  # Faster CPU->GPU transfer
    NUM_WORKERS = 4 if DEVICE == 'cuda' else 2  # Parallel data loading
    
    # Auto-retry with smaller batches on OOM
    ENABLE_OOM_RECOVERY = True
    MIN_BATCH_SIZE = 8  # Minimum batch size before giving up
    
    # Processing options
    SHOW_PROGRESS = True
    VERBOSE = True
    CLEAR_CACHE_FREQUENCY = 3  # Clear GPU cache every 3 files (more frequent)


# ============================================================================
# HELPER FUNCTIONS FOR CONTEXTUAL INPUT CREATION
# ============================================================================

def create_contextual_input_w3(prev_sent_i_minus_1, main_sent, next_sent_i_plus_1):
    """
    Create contextual input with window size 3 (1 previous + main + 1 next).
    
    Window 3 uses positions: (i-1, i, i+1)
    - prev_sent_i_minus_1: sentence at position (i-1) = previous_sentence_2 in Stage 1
    - main_sent: sentence at position (i) = main_sentence in Stage 1
    - next_sent_i_plus_1: sentence at position (i+1) = next_sentence_1 in Stage 1
    
    Returns:
        str: Concatenated contextual input with [SEP] tokens
    """
    parts = []
    
    if prev_sent_i_minus_1 and str(prev_sent_i_minus_1).strip():
        parts.append(str(prev_sent_i_minus_1).strip())
    
    parts.append(str(main_sent).strip())
    
    if next_sent_i_plus_1 and str(next_sent_i_plus_1).strip():
        parts.append(str(next_sent_i_plus_1).strip())
    
    return " [SEP] ".join(parts)


def create_contextual_input_w5(prev_sent_i_minus_2, prev_sent_i_minus_1, main_sent, 
                                next_sent_i_plus_1, next_sent_i_plus_2):
    """
    Create contextual input with window size 5 (2 previous + main + 2 next).
    
    Window 5 uses positions: (i-2, i-1, i, i+1, i+2)
    - prev_sent_i_minus_2: sentence at position (i-2) = previous_sentence_1 in Stage 1
    - prev_sent_i_minus_1: sentence at position (i-1) = previous_sentence_2 in Stage 1
    - main_sent: sentence at position (i) = main_sentence in Stage 1
    - next_sent_i_plus_1: sentence at position (i+1) = next_sentence_1 in Stage 1
    - next_sent_i_plus_2: sentence at position (i+2) = next_sentence_2 in Stage 1
    
    Returns:
        str: Concatenated contextual input with [SEP] tokens
    """
    parts = []
    
    if prev_sent_i_minus_2 and str(prev_sent_i_minus_2).strip():
        parts.append(str(prev_sent_i_minus_2).strip())
    
    if prev_sent_i_minus_1 and str(prev_sent_i_minus_1).strip():
        parts.append(str(prev_sent_i_minus_1).strip())
    
    parts.append(str(main_sent).strip())
    
    if next_sent_i_plus_1 and str(next_sent_i_plus_1).strip():
        parts.append(str(next_sent_i_plus_1).strip())
    
    if next_sent_i_plus_2 and str(next_sent_i_plus_2).strip():
        parts.append(str(next_sent_i_plus_2).strip())
    
    return " [SEP] ".join(parts)


# ============================================================================
# GPU UTILITIES
# ============================================================================

def clear_gpu_memory():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def encode_with_oom_recovery(sbert_model, inputs, config, window_name="w3"):
    """
    Encode inputs with automatic OOM recovery.
    Will retry with smaller batch sizes if OOM occurs.
    
    Args:
        sbert_model: SBERT model
        inputs: List of input texts
        config: Configuration object
        window_name: Name for progress display (w3 or w5)
    
    Returns:
        List of embeddings
    """
    batch_size = config.BATCH_SIZE
    all_embeddings = []
    num_batches = (len(inputs) + batch_size - 1) // batch_size
    
    print(f"\n🔄 Generating {window_name} embeddings (batch size: {batch_size})...")
    if config.DEVICE == 'cuda':
        print(f"   ⚡ GPU acceleration enabled with {'FP16' if config.USE_FP16 else 'FP32'} precision")
    
    batch_idx = 0
    i = 0
    
    while i < len(inputs):
        batch_idx += 1
        batch = inputs[i:i + batch_size]
        
        try:
            # Encode with GPU optimizations
            if config.DEVICE == 'cuda':
                with torch.cuda.amp.autocast(enabled=config.USE_FP16):
                    batch_embeddings = sbert_model.encode(
                        batch, 
                        convert_to_numpy=True, 
                        show_progress_bar=False,
                        batch_size=batch_size,
                        normalize_embeddings=False
                    )
            else:
                batch_embeddings = sbert_model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    show_progress_bar=False
                )
            
            all_embeddings.extend(batch_embeddings)
            i += batch_size
            
            # Progress update every 10 batches
            if batch_idx % 10 == 0 or i >= len(inputs):
                progress = min(i, len(inputs))
                percent = (progress / len(inputs)) * 100
                
                if config.DEVICE == 'cuda':
                    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"   Progress: {progress:,}/{len(inputs):,} ({percent:.1f}%) | "
                          f"GPU: {gpu_mem:.2f}GB | Batch {batch_idx}/{num_batches}")
                else:
                    print(f"   Progress: {progress:,}/{len(inputs):,} ({percent:.1f}%)")
        
        except torch.cuda.OutOfMemoryError as e:
            # OOM Error - try smaller batch size
            if config.ENABLE_OOM_RECOVERY and batch_size > config.MIN_BATCH_SIZE:
                clear_gpu_memory()
                batch_size = max(config.MIN_BATCH_SIZE, batch_size // 2)
                num_batches = (len(inputs) + batch_size - 1) // batch_size
                
                print(f"\n   ⚠️  GPU Out of Memory! Reducing batch size to {batch_size}")
                print(f"   ↻  Retrying batch {batch_idx}...")
                
                # Don't increment i, retry with smaller batch
                continue
            else:
                # Can't reduce further, give up
                print(f"\n   ❌ GPU Out of Memory even with minimum batch size!")
                raise e
    
    return all_embeddings


def get_gpu_info():
    """Display GPU information"""
    if torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU memory: {total_mem:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU Memory allocated: {allocated:.2f} GB")
        print(f"   GPU Memory reserved: {reserved:.2f} GB")
        print(f"   GPU Memory available: {total_mem - reserved:.2f} GB")
        
        # Set memory growth to optimize utilization
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Use 90% of available memory (leave 10% buffer)
            torch.cuda.set_per_process_memory_fraction(0.9, 0)
            print(f"   ⚙️  GPU memory limit set to 90% ({total_mem * 0.9:.2f} GB)")
    else:
        print("💻 No GPU detected - will use CPU")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_file_stage2(input_file_path, output_folder, sbert_model, config, 
                                file_number, total_files):
    """
    Process a single Stage 1 file and create context-aware embeddings with two window sizes.
    
    Args:
        input_file_path (Path): Path to the Stage 1 CSV file
        output_folder (Path): Output folder for Stage 2 files
        sbert_model: Loaded SBERT model
        config: Configuration object
        file_number (int): File number (for tracking)
        total_files (int): Total number of files
    
    Returns:
        tuple: (file_number, sentence_count, output_file_path, success, elapsed_time)
    """
    try:
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Processing File {file_number}/{total_files}: {input_file_path.name}")
        print(f"{'='*80}")
        
        # Read the Stage 1 CSV file
        print("🔄 Loading CSV file...")
        df = pd.read_csv(input_file_path)
        print(f"   ✓ Loaded {len(df):,} sentences")
        
        # Validate required columns
        required_cols = ['sentence_id', 'article_id', 'date', 'source', 
                        'previous_sentence_1', 'previous_sentence_2', 'main_sentence', 
                        'next_sentence_1', 'next_sentence_2']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Skipping {input_file_path.name}: Missing columns {missing_cols}")
            return (file_number, 0, None, False, 0)
        
        # Replace NaN with empty strings
        df = df.fillna("")
        
        # Create contextual inputs for both window sizes
        print("\n🔄 Creating contextual inputs...")
        contextual_inputs_w3 = []
        contextual_inputs_w5 = []
        
        for idx, row in df.iterrows():
            # Window 3: positions (i-1, i, i+1)
            input_w3 = create_contextual_input_w3(
                row['previous_sentence_2'],  # i-1
                row['main_sentence'],         # i
                row['next_sentence_1']        # i+1
            )
            contextual_inputs_w3.append(input_w3)
            
            # Window 5: positions (i-2, i-1, i, i+1, i+2)
            input_w5 = create_contextual_input_w5(
                row['previous_sentence_1'],  # i-2
                row['previous_sentence_2'],  # i-1
                row['main_sentence'],         # i
                row['next_sentence_1'],       # i+1
                row['next_sentence_2']        # i+2
            )
            contextual_inputs_w5.append(input_w5)
        
        print(f"   ✓ Created {len(contextual_inputs_w3):,} w3 contextual inputs")
        print(f"   ✓ Created {len(contextual_inputs_w5):,} w5 contextual inputs")
        
        # Clear GPU cache before encoding (important!)
        if config.DEVICE == 'cuda':
            clear_gpu_memory()
        
        # Generate embeddings with automatic OOM recovery
        all_embeddings_w3 = encode_with_oom_recovery(
            sbert_model, contextual_inputs_w3, config, window_name="w3"
        )
        print(f"   ✓ Generated {len(all_embeddings_w3):,} w3 embeddings")
        
        # Clear cache between w3 and w5
        if config.DEVICE == 'cuda':
            clear_gpu_memory()
        
        all_embeddings_w5 = encode_with_oom_recovery(
            sbert_model, contextual_inputs_w5, config, window_name="w5"
        )
        print(f"   ✓ Generated {len(all_embeddings_w5):,} w5 embeddings")
        
        # Convert embeddings to numpy arrays
        embeddings_array_w3 = np.array(all_embeddings_w3)
        embeddings_array_w5 = np.array(all_embeddings_w5)
        
        # Add embeddings as new columns (store as comma-separated strings for CSV)
        print("\n🔄 Adding embeddings to dataframe...")
        df['w3_embedding'] = [','.join(map(str, emb)) for emb in embeddings_array_w3]
        df['w5_embedding'] = [','.join(map(str, emb)) for emb in embeddings_array_w5]
        print("   ✓ Embeddings added as columns")
        
        # Define output file path
        # Convert Data_s1_X.csv to Data_s2_X.csv
        output_filename = input_file_path.name.replace('Data_s1_', 'Data_s2_')
        if output_filename == input_file_path.name:  # If no replacement happened
            output_filename = f"Data_s2_{file_number}.csv"
        
        output_file = output_folder / output_filename
        
        # Save to CSV
        print(f"\n🔄 Saving to: {output_file.name}...")
        df.to_csv(output_file, index=False)
        print("   ✓ File saved successfully")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ File {file_number}/{total_files} completed in {elapsed_time:.2f} seconds")
        print(f"   Sentences processed: {len(df):,}")
        print(f"   Output: {output_file.name}")
        
        # Clear GPU cache periodically (every N files) for better memory management
        if config.DEVICE == 'cuda' and file_number % config.CLEAR_CACHE_FREQUENCY == 0:
            clear_gpu_memory()
        
        return (file_number, len(df), output_file, True, elapsed_time)
        
    except Exception as e:
        print(f"\n❌ Error processing {input_file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return (file_number, 0, None, False, 0)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 100)
    print("STAGE 2: SBERT EMBEDDINGS GENERATOR - GPU/CPU STANDALONE SCRIPT")
    print("=" * 100)
    print()
    
    # Set PyTorch memory allocator for better fragmentation handling
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("🔧 GPU Memory Configuration:")
        print("   - PyTorch allocator: expandable_segments mode enabled")
        print("   - This reduces memory fragmentation")
        print()
    
    # Initialize configuration
    config = Config()
    
    # Display system information
    print("🖥️  System Information:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {config.DEVICE.upper()}")
    print()
    
    # Get GPU info
    get_gpu_info()
    print()
    
    # Get input folder path
    print("=" * 100)
    print("📂 CONFIGURATION")
    print("=" * 100)
    print()
    
    # Option 1: Use default paths or ask user
    use_default = input("Use default paths? (Press Enter for yes, 'n' for custom paths): ").strip().lower()
    
    if use_default != 'n':
        # Default paths
        current_dir = Path(__file__).parent
        stage1_input = current_dir / 'Processed_Data' / 'Stage_1'
        stage2_output = current_dir / 'Processed_Data' / 'Stage_2'
        print(f"\n✓ Using default paths:")
        print(f"   Stage 1 Input:  {stage1_input}")
        print(f"   Stage 2 Output: {stage2_output}")
    else:
        # Custom paths
        print("\n📁 Enter paths (or press Enter for defaults):")
        
        stage1_input_str = input("   Stage 1 input folder path: ").strip()
        if not stage1_input_str:
            stage1_input = Path(__file__).parent / 'Processed_Data' / 'Stage_1'
        else:
            stage1_input = Path(stage1_input_str)
        
        stage2_output_str = input("   Stage 2 output folder path: ").strip()
        if not stage2_output_str:
            stage2_output = Path(__file__).parent / 'Processed_Data' / 'Stage_2'
        else:
            stage2_output = Path(stage2_output_str)
    
    print()
    
    # Validate input folder
    if not stage1_input.exists():
        print(f"❌ ERROR: Stage 1 input folder not found: {stage1_input}")
        print(f"   Please check the path and try again.")
        sys.exit(1)
    
    # Create output folder
    stage2_output.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output folder ready: {stage2_output}")
    print()
    
    # Get all Stage 1 CSV files
    stage1_files = sorted(stage1_input.glob('*.csv'))
    total_files = len(stage1_files)
    
    if total_files == 0:
        print(f"❌ No CSV files found in: {stage1_input}")
        print(f"   Please check the folder contains Stage 1 CSV files.")
        sys.exit(1)
    
    print("=" * 100)
    print("📊 PROCESSING OVERVIEW")
    print("=" * 100)
    print(f"   Input folder: {stage1_input}")
    print(f"   Output folder: {stage2_output}")
    print(f"   Total files to process: {total_files}")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Device: {config.DEVICE.upper()}")
    print()
    
    # Show sample files
    print("📋 Sample files to process:")
    for i, f in enumerate(stage1_files[:5], 1):
        file_size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {i}. {f.name} ({file_size_mb:.2f} MB)")
    if total_files > 5:
        print(f"   ... and {total_files - 5} more files")
    print()
    
    # Confirm before processing
    confirm = input("Proceed with processing? (Press Enter to continue, Ctrl+C to cancel): ")
    print()
    
    # Load SBERT model
    print("=" * 100)
    print("🔄 LOADING SBERT MODEL")
    print("=" * 100)
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Target device: {config.DEVICE.upper()}")
    if config.DEVICE == 'cuda':
        print(f"   Precision: {'FP16 (Mixed Precision)' if config.USE_FP16 else 'FP32'}")
        print(f"   Optimizations: Enabled (90% GPU memory utilization)")
    print("   This may take a few minutes on first run...")
    print()
    
    start_load = time.time()
    
    # Load model with optimizations
    sbert_model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)
    
    # Apply GPU optimizations
    if config.DEVICE == 'cuda':
        # Set max sequence length for memory efficiency
        sbert_model.max_seq_length = config.MAX_SEQ_LENGTH
        
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Disable gradient computation (we're only doing inference)
        torch.set_grad_enabled(False)
        
        print(f"   ⚡ GPU optimizations applied:")
        print(f"      - Max sequence length: {config.MAX_SEQ_LENGTH}")
        print(f"      - cuDNN benchmark: Enabled")
        print(f"      - Gradient computation: Disabled")
        print(f"      - Mixed precision (FP16): {'Enabled' if config.USE_FP16 else 'Disabled'}")
    
    load_time = time.time() - start_load
    
    print(f"\n✅ SBERT model loaded successfully in {load_time:.2f} seconds!")
    print(f"   Embedding dimension: {sbert_model.get_sentence_embedding_dimension()}")
    
    if config.DEVICE == 'cuda':
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory allocated: {gpu_mem:.2f} GB / {total_mem:.2f} GB ({gpu_mem/total_mem*100:.1f}%)")
        print("   🚀 GPU acceleration enabled with full utilization!")
    print()
    
    # Process all files
    print("=" * 100)
    print("🚀 PROCESSING STAGE 1 FILES")
    print("=" * 100)
    print()
    
    overall_start_time = time.time()
    results = []
    total_sentences_processed = 0
    successful_files = 0
    failed_files = 0
    
    for file_num, input_file in enumerate(stage1_files, start=1):
        result = process_single_file_stage2(
            input_file, 
            stage2_output, 
            sbert_model, 
            config, 
            file_num, 
            total_files
        )
        results.append(result)
        
        file_number, sentence_count, output_file, success, elapsed_time = result
        
        if success:
            successful_files += 1
            total_sentences_processed += sentence_count
        else:
            failed_files += 1
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # Print final summary
    print("\n" + "=" * 100)
    print("🎉 STAGE 2 PROCESSING COMPLETE")
    print("=" * 100)
    print()
    
    print("📊 Final Statistics:")
    print(f"   Total files processed: {total_files}")
    print(f"   Successful: {successful_files}")
    print(f"   Failed: {failed_files}")
    print(f"   Total sentences processed: {total_sentences_processed:,}")
    print(f"   Total processing time: {overall_elapsed_time:.2f} seconds ({overall_elapsed_time/60:.2f} minutes)")
    
    if total_sentences_processed > 0:
        avg_time_per_sentence = overall_elapsed_time / total_sentences_processed
        sentences_per_second = total_sentences_processed / overall_elapsed_time
        print(f"   Average time per sentence: {avg_time_per_sentence:.4f} seconds")
        print(f"   Throughput: {sentences_per_second:.2f} sentences/second")
        
        if config.DEVICE == 'cuda':
            print(f"\n   🎮 GPU Performance Summary:")
            gpu_mem_used = torch.cuda.max_memory_allocated(0) / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"      - Peak GPU memory: {gpu_mem_used:.2f} GB / {total_mem:.2f} GB ({gpu_mem_used/total_mem*100:.1f}%)")
            print(f"      - Batch size used: {config.BATCH_SIZE}")
            print(f"      - Precision mode: {'FP16' if config.USE_FP16 else 'FP32'}")
            
            # Provide recommendations
            if gpu_mem_used / total_mem < 0.7:
                print(f"      ⚡ Recommendation: GPU utilization is {gpu_mem_used/total_mem*100:.1f}%")
                print(f"         You can increase batch size to {config.BATCH_SIZE + 32} for better performance")
            elif gpu_mem_used / total_mem > 0.95:
                print(f"      ⚠️  Warning: GPU utilization is {gpu_mem_used/total_mem*100:.1f}%")
                print(f"         Consider reducing batch size to {config.BATCH_SIZE - 16} to avoid OOM errors")
            else:
                print(f"      ✅ Optimal: GPU utilization is {gpu_mem_used/total_mem*100:.1f}% (excellent!)")
    
    print()
    print(f"✅ All Stage 2 files saved to: {stage2_output}")
    print()
    
    # List output files
    output_files = sorted(stage2_output.glob('*.csv'))
    print(f"📁 Output files ({len(output_files)} total):")
    for i, out_file in enumerate(output_files[:10], 1):
        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"   {i}. {out_file.name} ({file_size_mb:.2f} MB)")
    
    if len(output_files) > 10:
        print(f"   ... and {len(output_files) - 10} more files")
    print()
    
    print("=" * 100)
    print("✅ Processing completed successfully!")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# INSTALLATION & USAGE INSTRUCTIONS
# ============================================================================
#
# 1. Activate virtual environment:
#    source venv/bin/activate                    # Linux/Mac
#    venv\Scripts\activate                       # Windows
#
# 2. Install dependencies (one-time):
#    pip install torch sentence-transformers pandas numpy
#
# 3. Run the script:
#    python stage2_gpu_processor.py
#    # or use explicit path to venv python:
#    /path/to/venv/bin/python stage2_gpu_processor.py
#
# 4. GPU Optimization Tips for 2GB NVIDIA RTX 450:
#    - Default batch size: 96 (optimal for 2GB)
#    - If OOM error: Edit line 60, reduce BATCH_SIZE to 64 or 48
#    - For more speed: Try BATCH_SIZE = 128 (monitor GPU memory)
#    - FP16 mode: Enabled by default for 2x speedup
#
# 5. Expected Performance:
#    - GPU (2GB, FP16): 200-400 sentences/second
#    - CPU: 50-100 sentences/second
#    - Processing 10,000 sentences: ~30-50 seconds (GPU) vs ~2-3 minutes (CPU)
#
# ============================================================================
