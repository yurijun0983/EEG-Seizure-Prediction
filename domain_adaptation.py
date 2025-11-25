"""
Step 3: Domain Adaptation - Align all patients' data to the anchor patient
Uses Euclidean Alignment (EA) - a proven method for EEG cross-subject transfer

Reference:
- "Revisiting Euclidean alignment for transfer learning in EEG" (2025)
- "Domain adaptation for epileptic EEG classification" (2022)
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
        # Compute covariance for this segment
        cov = np.cov(seg)  # (channels, channels)
        cov_sum += cov
    
    # Average covariance
    cov_avg = cov_sum / len(segments)
    return cov_avg


def euclidean_alignment_transform(cov_source, cov_target):
    """
    Compute Euclidean Alignment transformation matrix
    
    Args:
        cov_source: Source domain covariance matrix (channels, channels)
        cov_target: Target domain covariance matrix (channels, channels)
    
    Returns:
        Transformation matrix to align source to target
    """
    # Compute matrix square roots using eigendecomposition
    # cov_target^{-1/2}
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_target)
    eigvals_t = np.maximum(eigvals_t, 1e-8)  # Avoid numerical issues
    cov_target_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(eigvals_t)) @ eigvecs_t.T
    
    # cov_source^{1/2}
    eigvals_s, eigvecs_s = np.linalg.eigh(cov_source)
    eigvals_s = np.maximum(eigvals_s, 1e-8)
    cov_source_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
    
    # Transformation matrix: R = cov_target^{-1/2} @ cov_source^{1/2}
    R = cov_target_inv_sqrt @ cov_source_sqrt
    
    return R


def apply_euclidean_alignment(segments, transform_matrix):
    """
    Apply Euclidean Alignment transformation to segments
    
    Args:
        segments: List of EEG segments, each (channels, samples)
        transform_matrix: Transformation matrix (channels, channels)
    
    Returns:
        List of aligned segments
    """
    aligned_segments = []
    
    for seg in segments:
        # Apply transformation: X_aligned = R @ X_source
        seg_aligned = transform_matrix @ seg
        aligned_segments.append(seg_aligned.astype(np.float32))
    
    return aligned_segments


def align_patient_to_anchor(
    source_patient_id,
    anchor_patient_id,
    preprocessed_dir,
    output_dir
):
    """
    Align a source patient's data to the anchor patient using Euclidean Alignment
    
    Args:
        source_patient_id: Patient ID to align
        anchor_patient_id: Anchor (target) patient ID
        preprocessed_dir: Directory with original preprocessed data
        output_dir: Directory to save aligned data
    
    Returns:
        Statistics about the alignment
    """
    print(f"\nAligning {source_patient_id} → {anchor_patient_id}")
    
    # Load source patient data
    source_file = os.path.join(preprocessed_dir, f'{source_patient_id}.npz')
    source_data = np.load(source_file)
    source_preictal = list(source_data['preictal'])
    source_interictal = list(source_data['interictal'])
    source_data.close()
    
    # Load anchor patient data
    anchor_file = os.path.join(preprocessed_dir, f'{anchor_patient_id}.npz')
    anchor_data = np.load(anchor_file)
    anchor_preictal = list(anchor_data['preictal'])
    anchor_interictal = list(anchor_data['interictal'])
    anchor_data.close()
    
    # Compute covariance matrices
    print("  Computing covariance matrices...")
    source_all = source_preictal + source_interictal
    anchor_all = anchor_preictal + anchor_interictal
    
    cov_source = compute_covariance_matrix(source_all)
    cov_anchor = compute_covariance_matrix(anchor_all)
    
    # Compute transformation
    print("  Computing Euclidean Alignment transformation...")
    transform_matrix = euclidean_alignment_transform(cov_source, cov_anchor)
    
    # Apply transformation
    print("  Applying transformation to segments...")
    aligned_preictal = apply_euclidean_alignment(source_preictal, transform_matrix)
    aligned_interictal = apply_euclidean_alignment(source_interictal, transform_matrix)
    
    # Save aligned data
    output_file = os.path.join(output_dir, f'{source_patient_id}.npz')
    np.savez_compressed(
        output_file,
        preictal=np.array(aligned_preictal, dtype=np.float32),
        interictal=np.array(aligned_interictal, dtype=np.float32)
    )
    
    print(f"  Saved aligned data to {output_file}")
    
    # Compute statistics
    stats = {
        'source_patient': source_patient_id,
        'anchor_patient': anchor_patient_id,
        'n_preictal': len(aligned_preictal),
        'n_interictal': len(aligned_interictal),
        'transform_matrix_norm': float(np.linalg.norm(transform_matrix))
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Domain Adaptation: Align all patients to anchor patient'
    )
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Directory with original preprocessed .npz files')
    parser.add_argument('--anchor_patient', type=str, required=True,
                       help='Anchor patient ID (best source model)')
    parser.add_argument('--test_patient', type=str, required=True,
                       help='Test patient ID (excluded from alignment)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_aligned',
                       help='Output directory for aligned data')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset (to get patient list)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all patient IDs
    all_patients = sorted([d for d in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
    
    # Remove test patient
    source_patients = [p for p in all_patients if p != args.test_patient]
    
    print(f"\n{'='*60}")
    print("Domain Adaptation: Euclidean Alignment")
    print(f"{'='*60}")
    print(f"Anchor patient: {args.anchor_patient}")
    print(f"Test patient: {args.test_patient}")
    print(f"Source patients to align: {source_patients}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Process each source patient
    all_stats = {}
    
    for patient_id in source_patients:
        try:
            if patient_id == args.anchor_patient:
                # Copy anchor patient data without modification
                print(f"\n{patient_id} is the anchor - copying without alignment...")
                source_file = os.path.join(args.preprocessed_dir, f'{patient_id}.npz')
                output_file = os.path.join(args.output_dir, f'{patient_id}.npz')
                
                data = np.load(source_file)
                np.savez_compressed(
                    output_file,
                    preictal=data['preictal'],
                    interictal=data['interictal']
                )
                data.close()
                
                all_stats[patient_id] = {
                    'is_anchor': True,
                    'n_preictal': len(np.load(output_file)['preictal']),
                    'n_interictal': len(np.load(output_file)['interictal'])
                }
                np.load(output_file).close()
            else:
                # Align to anchor
                stats = align_patient_to_anchor(
                    source_patient_id=patient_id,
                    anchor_patient_id=args.anchor_patient,
                    preprocessed_dir=args.preprocessed_dir,
                    output_dir=args.output_dir
                )
                all_stats[patient_id] = stats
        
        except Exception as e:
            print(f"ERROR processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save alignment summary
    summary = {
        'anchor_patient': args.anchor_patient,
        'test_patient': args.test_patient,
        'aligned_patients': list(all_stats.keys()),
        'statistics': all_stats,
        'args': vars(args)
    }
    
    summary_path = os.path.join(args.output_dir, 'alignment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Domain Adaptation Complete!")
    print(f"{'='*60}")
    print(f"Aligned data saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}\n")


if __name__ == '__main__':
    main()
