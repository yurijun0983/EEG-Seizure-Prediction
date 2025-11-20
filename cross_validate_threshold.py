"""
Cross-validation based threshold selection for robust threshold optimization
Uses k-fold cross-validation to find the optimal decision threshold
"""

import os
import argparse
import json
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import KFold
from tqdm import tqdm

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import load_single_patient_dataloader
from torch.utils.data import DataLoader, Subset


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': 0.0,
        'sensitivity': 0.0,
        'auc': 0.0
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
        
        if (tp + fn) > 0:
            metrics['sensitivity'] = tp / (tp + fn)
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    return metrics


def evaluate_threshold_on_fold(model, data_loader, device, threshold):
    """Evaluate a single threshold on one fold"""
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Apply threshold
    preds = (all_probs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, preds, all_probs)
    
    return metrics


def cross_validate_threshold(model, full_dataset, device, thresholds, n_folds=5, batch_size=32):
    """
    Use k-fold cross-validation to evaluate thresholds
    
    Returns:
        threshold_scores: Dict mapping threshold -> list of fold scores
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    threshold_scores = {t: [] for t in thresholds}
    
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    print("="*80)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        print("-"*80)
        
        # Create validation subset
        val_subset = Subset(full_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Validation samples: {len(val_subset)}")
        
        # Evaluate each threshold on this fold
        for threshold in thresholds:
            metrics = evaluate_threshold_on_fold(model, val_loader, device, threshold)
            threshold_scores[threshold].append(metrics['f1'])  # Use F1 as primary metric
    
    return threshold_scores


def find_optimal_threshold_cv(threshold_scores, metric='f1'):
    """
    Find threshold with best average performance across folds
    
    Returns:
        best_threshold: Optimal threshold
        mean_score: Mean score across folds
        std_score: Standard deviation across folds
    """
    threshold_stats = {}
    
    for threshold, scores in threshold_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold_stats[threshold] = {
            'mean': mean_score,
            'std': std_score,
            'scores': scores
        }
    
    # Find threshold with highest mean score
    best_threshold = max(threshold_stats.items(), key=lambda x: x[1]['mean'])
    
    return best_threshold[0], best_threshold[1]['mean'], best_threshold[1]['std'], threshold_stats


def main():
    parser = argparse.ArgumentParser(description='Cross-validation based threshold selection')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files')
    
    # Data configuration
    parser.add_argument('--val_patient', type=str, required=True,
                       help='Patient ID for validation (will use its data for CV)')
    parser.add_argument('--window_seconds', type=float, default=10.0,
                       help='Window size in seconds')
    parser.add_argument('--overlap_seconds', type=float, default=5.0,
                       help='Overlap in seconds')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated thresholds (e.g., "0.3,0.4,0.5")')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Parse thresholds
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        thresholds = np.arange(0.1, 1.0, 0.05).tolist()
    
    print("\n" + "="*60)
    print("Cross-Validation Threshold Selection")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Validation patient: {args.val_patient}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Thresholds to evaluate: {len(thresholds)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data - use test_loader which contains the specified patient
    print("="*60)
    print("Loading patient data...")
    print("="*60)
    
    patient_loader = load_single_patient_dataloader(
        data_root=args.data_root,
        patient_id=args.val_patient,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        preprocessed_dir=args.preprocessed_dir
    )
    
    full_dataset = patient_loader.dataset
    
    # Get input dimensions
    sample_batch, _ = next(iter(patient_loader))
    num_channels = sample_batch.shape[1]
    sequence_length = sample_batch.shape[2]
    print(f"Input shape: (batch_size, {num_channels}, {sequence_length})\n")
    
    # Load model
    print("="*60)
    print("Loading model...")
    print("="*60)
    
    model_kwargs = {
        'num_channels': num_channels,
        'sequence_length': sequence_length,
        'num_classes': 2,
    }
    
    if args.model_type in ['full', 'simplified']:
        model_kwargs['use_binary_activation'] = False
    
    model = create_bsdcnn_model(model_type=args.model_type, **model_kwargs)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from: {args.model_path}\n")
    
    # Perform cross-validation
    threshold_scores = cross_validate_threshold(
        model, full_dataset, device, thresholds, 
        n_folds=args.n_folds, batch_size=args.batch_size
    )
    
    # Find optimal threshold
    best_threshold, mean_score, std_score, all_stats = find_optimal_threshold_cv(threshold_scores)
    
    # Print results
    print("\n" + "="*80)
    print("Cross-Validation Results")
    print("="*80)
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Mean F1 score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"\nFold-wise F1 scores: {all_stats[best_threshold]['scores']}")
    
    print(f"\n{'Threshold':>10} {'Mean F1':>10} {'Std F1':>10}")
    print("-"*32)
    for threshold in sorted(all_stats.keys()):
        stats = all_stats[threshold]
        print(f"{threshold:>10.4f} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = {
            'best_threshold': best_threshold,
            'best_mean_f1': mean_score,
            'best_std_f1': std_score,
            'n_folds': args.n_folds,
            'all_threshold_stats': {str(k): v for k, v in all_stats.items()},
            'model_path': args.model_path,
            'val_patient': args.val_patient
        }
        
        output_file = os.path.join(args.output_dir, 'cv_threshold_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*80)
    print("Cross-validation completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
