"""
Complete pipeline for domain adaptation-based epilepsy prediction

Pipeline:
1. Preprocess raw EEG data
2. Train individual models for each patient
3. Find best source model (anchor) for test patient
4. Align all patients' data to anchor using Euclidean Alignment
5. Train final model with aligned data
6. Evaluate on test patient (leave-one-out)
"""

import os
import argparse
import subprocess
import json


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
        description='Complete domain adaptation pipeline for epilepsy prediction'
    )
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of raw dataset')
    parser.add_argument('--test_patient', type=str, required=True,
                       help='Test patient ID (e.g., PN14)')
    
    # Pipeline control
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--skip_individual_training', action='store_true',
                       help='Skip individual model training if already done')
    parser.add_argument('--skip_find_anchor', action='store_true',
                       help='Skip finding anchor if already done')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='Skip domain adaptation alignment if already done')
    parser.add_argument('--anchor_patient', type=str, default=None,
                       help='Manually specify anchor patient (skip automatic finding)')
    
    # Training parameters
    parser.add_argument('--epochs_individual', type=int, default=50,
                       help='Epochs for individual model training')
    parser.add_argument('--epochs_final', type=int, default=100,
                       help='Epochs for final model training with aligned data')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    
    # Directories
    parser.add_argument('--preprocessed_dir', type=str, default='preprocessed_original',
                       help='Directory for original preprocessed data')
    parser.add_argument('--aligned_dir', type=str, default='preprocessed_aligned',
                       help='Directory for aligned preprocessed data')
    parser.add_argument('--individual_models_dir', type=str, default='outputs_individual_models',
                       help='Directory for individual patient models')
    parser.add_argument('--final_output_dir', type=str, default='outputs_final_model',
                       help='Directory for final model output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("COMPLETE DOMAIN ADAPTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Test patient: {args.test_patient}")
    print(f"Data root: {args.data_root}")
    print(f"Model type: {args.model_type}\n")
    
    # Step 1: Preprocess raw data
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
        print("\nSkipping Step 1: Preprocessing (using existing data)")
    
    # Step 2: Train individual models for each patient
    if not args.skip_individual_training:
        cmd = (
            f"python train_individual_models.py "
            f"--data_root {args.data_root} "
            f"--preprocessed_dir {args.preprocessed_dir} "
            f"--output_dir {args.individual_models_dir} "
            f"--epochs {args.epochs_individual} "
            f"--batch_size {args.batch_size} "
            f"--lr {args.lr} "
            f"--model_type {args.model_type}"
        )
        run_command(cmd, "Step 2: Training individual patient models")
    else:
        print("\nSkipping Step 2: Individual training (using existing models)")
    
    # Step 3: Find best source model (anchor)
    if args.anchor_patient:
        anchor_patient = args.anchor_patient
        print(f"\nUsing manually specified anchor patient: {anchor_patient}")
    elif not args.skip_find_anchor:
        cmd = (
            f"python find_best_source_model.py "
            f"--data_root {args.data_root} "
            f"--preprocessed_dir {args.preprocessed_dir} "
            f"--models_dir {args.individual_models_dir} "
            f"--test_patient {args.test_patient} "
            f"--batch_size {args.batch_size} "
            f"--model_type {args.model_type}"
        )
        run_command(cmd, "Step 3: Finding best source model (anchor)")
        
        # Load anchor from results
        results_file = os.path.join(args.individual_models_dir, 'best_source_model.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        anchor_patient = results['anchor_patient']
        print(f"\nAnchor patient identified: {anchor_patient}")
    else:
        # Load from existing results
        results_file = os.path.join(args.individual_models_dir, 'best_source_model.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        anchor_patient = results['anchor_patient']
        print(f"\nSkipping Step 3: Using existing anchor patient: {anchor_patient}")
    
    # Step 4: Domain adaptation - align all data to anchor
    if not args.skip_alignment:
        cmd = (
            f"python domain_adaptation.py "
            f"--preprocessed_dir {args.preprocessed_dir} "
            f"--anchor_patient {anchor_patient} "
            f"--test_patient {args.test_patient} "
            f"--output_dir {args.aligned_dir} "
            f"--data_root {args.data_root}"
        )
        run_command(cmd, "Step 4: Domain adaptation (Euclidean Alignment)")
    else:
        print("\nSkipping Step 4: Domain adaptation (using existing aligned data)")
    
    # Step 5: Train final model with aligned data
    cmd = (
        f"python bsdcnn_train.py "
        f"--data_root {args.data_root} "
        f"--preprocessed_dir {args.aligned_dir} "
        f"--test_patient {args.test_patient} "
        f"--output_dir {args.final_output_dir} "
        f"--epochs {args.epochs_final} "
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
    run_command(cmd, "Step 5: Training final model with aligned data")
    
    # Step 6: Test on test patient
    cmd = (
        f"python bsdcnn_test.py "
        f"--model_path {args.final_output_dir}/best_model.pth "
        f"--data_root {args.data_root} "
        f"--preprocessed_dir {args.preprocessed_dir} "
        f"--test_patient {args.test_patient} "
        f"--batch_size {args.batch_size} "
        f"--model_type {args.model_type} "
        f"--window_seconds 5 "
        f"--overlap_seconds 2.5 "
        f"--output_dir {args.final_output_dir}"
    )
    run_command(cmd, "Step 6: Final evaluation on test patient")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  - Individual models: {args.individual_models_dir}")
    print(f"  - Anchor patient: {anchor_patient}")
    print(f"  - Aligned data: {args.aligned_dir}")
    print(f"  - Final model: {args.final_output_dir}")
    print(f"  - Test results: {args.final_output_dir}/test_results.json\n")


if __name__ == '__main__':
    main()
