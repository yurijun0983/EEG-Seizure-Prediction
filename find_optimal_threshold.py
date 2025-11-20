"""
Find optimal threshold for BSDCNN model predictions
This script evaluates a trained model on the validation set using different thresholds
to find the one that maximizes F1 score or other metrics.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm
import matplotlib.pyplot as plt

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import load_single_patient_dataloader


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
        
        # Specificity (True Negative Rate)
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
        
        # Sensitivity (True Positive Rate, same as Recall)
        if (tp + fn) > 0:
            metrics['sensitivity'] = tp / (tp + fn)
        
        # Store confusion matrix values
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    # Calculate AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    return metrics


def evaluate_thresholds(model, val_loader, device, thresholds=None):
    """Evaluate model performance at different thresholds"""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)  # Default thresholds from 0.1 to 0.95
    
    model.eval()
    
    all_labels = []
    all_probs = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(val_loader, desc='Evaluating'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            
            # Get probabilities for positive class
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Evaluate each threshold
    results = []
    print(f"\nEvaluating {len(thresholds)} thresholds...")
    print("="*80)
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Specificity':>12} {'AUC':>10}")
    print("="*80)
    
    for threshold in thresholds:
        # Apply threshold
        preds = (all_probs >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, preds, all_probs)
        metrics['threshold'] = threshold
        
        results.append(metrics)
        
        # Print metrics for this threshold
        print(f"{threshold:>10.4f} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['specificity']:>12.4f} "
              f"{metrics['auc']:>10.4f}")
    
    print("="*80)
    return results, all_labels, all_probs


def find_optimal_threshold(results, metric='f1'):
    """Find the threshold that maximizes the specified metric"""
    if not results:
        return None, 0.0
    
    # Find the best result based on the specified metric
    best_result = max(results, key=lambda x: x[metric])
    return best_result['threshold'], best_result[metric]


def plot_threshold_performance(results, output_path=None):
    """Plot performance metrics vs threshold"""
    thresholds = [r['threshold'] for r in results]
    
    # Metrics to plot
    metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall', 'specificity']
    metric_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
    
    plt.figure(figsize=(12, 8))
    
    # Plot each metric
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        values = [r[metric] for r in results]
        plt.plot(thresholds, values, marker='o', label=name, linewidth=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Model Performance vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best F1 threshold line
    best_threshold, best_f1 = find_optimal_threshold(results, 'f1')
    plt.axvline(x=best_threshold, color='red', linestyle='--', 
                label=f'Best F1 Threshold ({best_threshold:.3f})')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Find optimal threshold for BSDCNN model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files')
    
    # Validation configuration
    parser.add_argument('--val_patient', type=str, required=True,
                       help='Patient ID for threshold evaluation (e.g., PN14)')
    parser.add_argument('--window_seconds', type=float, default=10.0,
                       help='Window size in seconds')
    parser.add_argument('--overlap_seconds', type=float, default=5.0,
                       help='Overlap in seconds')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Threshold search
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated list of thresholds to evaluate (e.g., "0.3,0.5,0.7")')
    parser.add_argument('--metric', type=str, default='f1',
                       choices=['f1', 'accuracy', 'precision', 'recall'],
                       help='Metric to optimize for threshold selection')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results and plots')
    
    args = parser.parse_args()
    
    # Parse custom thresholds if provided
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        thresholds = None
    
    # Print configuration
    print("\n" + "="*60)
    print("Optimal Threshold Finder Configuration")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Preprocessed dir: {args.preprocessed_dir}")
    print(f"Validation patient: {args.val_patient if args.val_patient else 'From training split'}")
    print(f"Window: {args.window_seconds}s, Overlap: {args.overlap_seconds}s")
    print(f"Model type: {args.model_type}")
    print(f"Optimization metric: {args.metric}")
    
    if thresholds:
        print(f"Custom thresholds: {thresholds}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data - only load the specified patient
    print("="*60)
    print("Loading test patient data...")
    print("="*60)
    
    if not args.val_patient:
        raise ValueError("--val_patient must be specified to indicate which patient to evaluate")
    
    test_loader = load_single_patient_dataloader(
        data_root=args.data_root,
        patient_id=args.val_patient,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        preprocessed_dir=args.preprocessed_dir
    )
    
    # Get input dimensions
    sample_batch, _ = next(iter(test_loader))
    num_channels = sample_batch.shape[1]
    sequence_length = sample_batch.shape[2]
    print(f"Input shape: (batch_size, {num_channels}, {sequence_length})\n")
    
    # Create model
    print("="*60)
    print("Loading model...")
    print("="*60)
    
    model_kwargs = {
        'num_channels': num_channels,
        'sequence_length': sequence_length,
        'num_classes': 2,
    }
    
    # Only add use_binary_activation for binary models
    if args.model_type in ['full', 'simplified']:
        model_kwargs['use_binary_activation'] = False
    
    model = create_bsdcnn_model(
        model_type=args.model_type,
        **model_kwargs
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from: {args.model_path}")
    if 'epoch' in checkpoint:
        print(f"Training epoch: {checkpoint['epoch']}")
    if 'val_f1' in checkpoint:
        print(f"Best validation F1: {checkpoint['val_f1']:.4f}\n")
    
    # Evaluate thresholds
    print("="*60)
    print("Finding optimal threshold...")
    print("="*60)
    
    results, labels, probs = evaluate_thresholds(model, test_loader, device, thresholds)
    
    # Find optimal threshold
    best_threshold, best_metric_value = find_optimal_threshold(results, args.metric)
    
    # Print results
    print("\n" + "="*60)
    print("Threshold Evaluation Results")
    print("="*60)
    print(f"Best threshold for {args.metric}: {best_threshold:.4f}")
    print(f"Best {args.metric} value: {best_metric_value:.4f}")
    
    # Print detailed results for best threshold
    best_result = next(r for r in results if abs(r['threshold'] - best_threshold) < 1e-6)
    print(f"\nDetailed metrics at best threshold ({best_threshold:.4f}):")
    print(f"  Accuracy:    {best_result['accuracy']:.4f}")
    print(f"  Precision:   {best_result['precision']:.4f}")
    print(f"  Recall:      {best_result['recall']:.4f}")
    print(f"  F1 Score:    {best_result['f1']:.4f}")
    print(f"  Specificity: {best_result['specificity']:.4f}")
    print(f"  Sensitivity: {best_result['sensitivity']:.4f}")
    print(f"  AUC:         {best_result['auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  TN: {best_result['true_negatives']:4d}  |  FP: {best_result['false_positives']:4d}")
    print(f"  FN: {best_result['false_negatives']:4d}  |  TP: {best_result['true_positives']:4d}")
    
    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(args.output_dir, 'threshold_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model_path': args.model_path,
                'metric_optimized': args.metric,
                'best_threshold': best_threshold,
                'best_metric_value': best_metric_value,
                'all_results': results
            }, f, indent=4)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Plot performance
        plot_file = os.path.join(args.output_dir, 'threshold_performance.png')
        plot_threshold_performance(results, plot_file)
    
    print("\n" + "="*60)
    print("Threshold optimization completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()