"""
Reverse Domain Adaptation: Align all source patients to the target (test) patient

Based on papers:
- "Calibration-free online test-time adaptation" (arXiv 2023)
- "Source-free unsupervised domain adaptation" (2024)
- "Target-driven domain adaptation for EEG" (Nature 2024)

Key idea: Instead of aligning target to source anchor, align ALL sources to the target.
This is especially useful when we have target data without labels.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import json


def compute_covariance_matrix(segments):
    """
    Compute average covariance matrix from EEG segments
    
    Args:
        segments: List of EEG segments, each with shape (channels, samples)
    
    Returns:
        Average covariance matrix (channels, channels)
    """
    n_channels = segments[0].shape[0]
    cov_sum = np.zeros((n_channels, n_channels))
    
    for seg in segments:
        cov = np.cov(seg)
        cov_sum += cov
    
    cov_avg = cov_sum / len(segments)
    return cov_avg


def reverse_euclidean_alignment(source_segments, target_cov):
    """
    Align source domain segments to target domain using Euclidean Alignment
    
    Args:
        source_segments: List of source EEG segments
        target_cov: Target domain covariance matrix (from test patient)
    
    Returns:
        List of aligned segments
    """
    # Compute source covariance
    source_cov = compute_covariance_matrix(source_segments)
    
    # Compute transformation: R = target_cov^{-1/2} @ source_cov^{1/2}
    eigvals_t, eigvecs_t = np.linalg.eigh(target_cov)
    eigvals_t = np.maximum(eigvals_t, 1e-8)
    target_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(eigvals_t)) @ eigvecs_t.T
    
    eigvals_s, eigvecs_s = np.linalg.eigh(source_cov)
    eigvals_s = np.maximum(eigvals_s, 1e-8)
    source_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
    
    transform = target_inv_sqrt @ source_sqrt
    
    # Apply transformation
    aligned = []
    for seg in source_segments:
        seg_aligned = transform @ seg
        aligned.append(seg_aligned.astype(np.float32))
    
    return aligned


def align_source_to_target(
    source_patient_id,
    target_patient_id,
    preprocessed_dir,
    output_dir,
    target_cov
):
    """
    Align a source patient's data to the target patient
    
    Args:
        source_patient_id: Source patient ID
        target_patient_id: Target (test) patient ID
        preprocessed_dir: Directory with original preprocessed data
        output_dir: Directory to save aligned data
        target_cov: Pre-computed target covariance matrix
    
    Returns:
        Statistics about the alignment
    """
    print(f"\nAligning {source_patient_id} → {target_patient_id} (REVERSE)")
    
    # Load source patient data
    source_file = os.path.join(preprocessed_dir, f'{source_patient_id}.npz')
    source_data = np.load(source_file)
    source_preictal = list(source_data['preictal'])
    source_interictal = list(source_data['interictal'])
    source_data.close()
    
    # Align to target
    print("  Applying reverse alignment...")
    aligned_preictal = reverse_euclidean_alignment(source_preictal, target_cov)
    aligned_interictal = reverse_euclidean_alignment(source_interictal, target_cov)
    
    # Save aligned data
    output_file = os.path.join(output_dir, f'{source_patient_id}.npz')
    np.savez_compressed(
        output_file,
        preictal=np.array(aligned_preictal, dtype=np.float32),
        interictal=np.array(aligned_interictal, dtype=np.float32)
    )
    
    print(f"  Saved to {output_file}")
    
    stats = {
        'source_patient': source_patient_id,
        'target_patient': target_patient_id,
        'n_preictal': len(aligned_preictal),
        'n_interictal': len(aligned_interictal),
        'alignment_type': 'reverse'
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Reverse Domain Adaptation: Align all sources to target patient'
    )
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Directory with original preprocessed .npz files')
    parser.add_argument('--target_patient', type=str, required=True,
                       help='Target (test) patient ID - used as alignment anchor')
    parser.add_argument('--output_dir', type=str, default='/data/preprocessed_reverse_aligned',
                       help='Output directory for aligned data')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset (to get patient list)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all patient IDs
    all_patients = sorted([d for d in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
    
    # Source patients (exclude target)
    source_patients = [p for p in all_patients if p != args.target_patient]
    
    print(f"\n{'='*70}")
    print("REVERSE Domain Adaptation: Align Sources to Target")
    print(f"{'='*70}")
    print(f"Target (anchor) patient: {args.target_patient}")
    print(f"Source patients to align: {source_patients}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load target patient data and compute covariance
    print(f"Computing target distribution from {args.target_patient}...")
    target_file = os.path.join(args.preprocessed_dir, f'{args.target_patient}.npz')
    target_data = np.load(target_file)
    target_all = list(target_data['preictal']) + list(target_data['interictal'])
    target_data.close()
    
    print(f"  Target has {len(target_all)} segments")
    print("  Computing target covariance matrix...")
    target_cov = compute_covariance_matrix(target_all)
    print(f"  Target covariance computed: shape {target_cov.shape}\n")
    
    # Process each source patient
    all_stats = {}
    
    for patient_id in source_patients:
        try:
            stats = align_source_to_target(
                source_patient_id=patient_id,
                target_patient_id=args.target_patient,
                preprocessed_dir=args.preprocessed_dir,
                output_dir=args.output_dir,
                target_cov=target_cov
            )
            all_stats[patient_id] = stats
        
        except Exception as e:
            print(f"ERROR processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Also align target data to itself for consistency
    print(f"\nAligning target patient {args.target_patient} to itself...")
    try:
        stats = align_source_to_target(
            source_patient_id=args.target_patient,
            target_patient_id=args.target_patient,
            preprocessed_dir=args.preprocessed_dir,
            output_dir=args.output_dir,
            target_cov=target_cov
        )
        all_stats[args.target_patient] = stats
    except Exception as e:
        print(f"ERROR processing target patient {args.target_patient}: {e}")
        import traceback
        traceback.print_exc()
    
    # Save alignment summary
    summary = {
        'target_patient': args.target_patient,
        'aligned_patients': list(all_stats.keys()),
        'alignment_type': 'reverse',
        'description': 'All source patients aligned TO target patient',
        'statistics': all_stats,
        'args': vars(args)
    }
    
    summary_path = os.path.join(args.output_dir, 'reverse_alignment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*70}")
    print("REVERSE Domain Adaptation Complete!")
    print(f"{'='*70}")
    print(f"All sources aligned to: {args.target_patient}")
    print(f"Aligned data saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}\n")


if __name__ == '__main__':
    main()
