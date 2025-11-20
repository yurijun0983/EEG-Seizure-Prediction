"""
Test script for BSDCNN seizure prediction model
Evaluates a trained model on test patients
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

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import create_bsdcnn_dataloaders


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


def test_model(model, test_loader, device, threshold=0.5):
    """Test the model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\nTesting with threshold={threshold}...")
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc='Testing'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            # Apply custom threshold
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description='Test BSDCNN model on test patients')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files')
    
    # Test configuration
    parser.add_argument('--test_patient', type=str, default=None,
                       help='Specific patient ID for testing (e.g., PN14)')
    parser.add_argument('--window_seconds', type=float, default=10.0,
                       help='Window size in seconds')
    parser.add_argument('--overlap_seconds', type=float, default=5.0,
                       help='Overlap in seconds')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for binary classification (default: 0.5)')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save test results (optional)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("Test Configuration")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Preprocessed dir: {args.preprocessed_dir}")
    print(f"Test patient: {args.test_patient if args.test_patient else 'All test patients'}")
    print(f"Window: {args.window_seconds}s, Overlap: {args.overlap_seconds}s")
    print(f"Model type: {args.model_type}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("="*60)
    print("Loading test data...")
    print("="*60)
    
    _, _, test_loader = create_bsdcnn_dataloaders(
        data_root=args.data_root,
        test_patient=args.test_patient,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        preprocessed_dir=args.preprocessed_dir,
        use_augmentation=False  # No augmentation for testing
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
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
    
    # Test model
    print("="*60)
    print("Testing model...")
    print("="*60)
    
    metrics, labels, preds, probs = test_model(model, test_loader, device, args.threshold)
    
    # Print results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:4d}  |  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  |  TP: {metrics['true_positives']:4d}")
    
    # Class distribution
    n_negative = sum([1 for l in labels if l == 0])
    n_positive = sum([1 for l in labels if l == 1])
    print(f"\nClass distribution:")
    print(f"  Interictal (0): {n_negative}")
    print(f"  Preictal (1):   {n_positive}")
    
    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = {
            'model_path': args.model_path,
            'test_patient': args.test_patient,
            'metrics': metrics,
            'class_distribution': {
                'interictal': n_negative,
                'preictal': n_positive
            }
        }
        
        output_file = os.path.join(args.output_dir, 'test_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
