"""
Complete Reverse Domain Adaptation Pipeline

Instead of finding best source and aligning to it,
we align ALL sources TO the target (test) patient.

This approach:
1. Uses test patient as the distribution anchor
2. Aligns all training data to match test distribution  
3. Trains model on aligned data
4. Evaluates on original test data

Benefits:
- Target-specific: Model specifically tuned for test patient
- Better distribution matching
- Supported by recent papers on test-time adaptation
"""

import os
import argparse
import subprocess


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed!")
        raise RuntimeError(f"Command failed: {cmd}")
    
    print(f"\n{description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Reverse Domain Adaptation Pipeline'
    )
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of raw dataset')
    parser.add_argument('--test_patient', type=str, required=True,
                       help='Test patient ID (used as alignment target)')
    
    # Pipeline control
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--preprocessed_dir', type=str, default='/data/preprocessed_original',
                       help='Directory for original preprocessed data')
    parser.add_argument('--reverse_aligned_dir', type=str, default='/data/preprocessed_reverse_aligned',
                       help='Directory for reverse-aligned data')
    parser.add_argument('--output_dir', type=str, default='/outputs_reverse_adaptation',
                       help='Directory for final model output')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("REVERSE DOMAIN ADAPTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Test patient (anchor): {args.test_patient}")
    print(f"Strategy: Align ALL sources TO test patient")
    print(f"Data root: {args.data_root}\n")
    
    # Step 1: Preprocess if needed
    if not args.skip_preprocess:
        cmd = (
            f"python bsdcnn_preprocess.py "
            f"--data_root {args.data_root} "
            f"--output_dir {args.preprocessed_dir} "
            f"--window_seconds 5 "
            f"--overlap_seconds 2.5"
        )
        run_command(cmd, "Step 1: Preprocessing raw EEG data")
    else:
        print("\nSkipping Step 1: Using existing preprocessed data")
    
    # Step 2: Reverse domain adaptation - align all to target
    cmd = (
        f"python reverse_domain_adaptation.py "
        f"--preprocessed_dir {args.preprocessed_dir} "
        f"--target_patient {args.test_patient} "
        f"--output_dir {args.reverse_aligned_dir} "
        f"--data_root {args.data_root}"
    )
    run_command(cmd, "Step 2: Reverse Domain Adaptation (Sources → Target)")
    
    # Step 3: Train model with reverse-aligned data
    cmd = (
        f"python bsdcnn_train.py "
        f"--data_root {args.data_root} "
        f"--preprocessed_dir {args.reverse_aligned_dir} "
        f"--test_patient {args.test_patient} "
        f"--output_dir {args.output_dir} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lr {args.lr} "
        f"--model_type {args.model_type} "
        f"--window_seconds 5 "
        f"--overlap_seconds 2.5 "
        f"--num_workers 0 "
        f"--use_focal_loss "
        f"--focal_alpha 0.75 "
        f"--focal_gamma 2.0"
    )
    run_command(cmd, "Step 3: Training with reverse-aligned data")
    
    # Step 4: Test on ORIGINAL test data (not aligned)
    cmd = (
        f"python bsdcnn_test.py "
        f"--model_path {args.output_dir}/best_model.pth "
        f"--data_root {args.data_root} "
        f"--preprocessed_dir {args.preprocessed_dir} "
        f"--test_patient {args.test_patient} "
        f"--batch_size {args.batch_size} "
        f"--model_type {args.model_type} "
        f"--window_seconds 5 "
        f"--overlap_seconds 2.5 "
        f"--threshold 0.5 "
        f"--output_dir {args.output_dir}"
    )
    run_command(cmd, "Step 4: Evaluation on ORIGINAL test data")
    
    print(f"\n{'='*70}")
    print("REVERSE ADAPTATION PIPELINE COMPLETED!")
    print(f"{'='*70}")
    print(f"\nKey difference from forward adaptation:")
    print(f"  - Forward: Sources aligned to BEST source")
    print(f"  - Reverse: Sources aligned to TEST patient")
    print(f"\nResults:")
    print(f"  - Aligned data: {args.reverse_aligned_dir}")
    print(f"  - Final model: {args.output_dir}")
    print(f"  - Test results: {args.output_dir}/test_results.json\n")


if __name__ == '__main__':
    main()
