"""
Evaluation script for BSDCNN (Binary Single-Dimensional CNN) for Seizure Prediction
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from bsdcnn_model import BSDCNN, BSDCNNSimplified, create_bsdcnn_model
from bsdcnn_data_loader import create_bsdcnn_dataloaders


def calculate_detailed_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Sensitivity (same as recall)
    sensitivity = recall
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.0
    
    # Additional metrics
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    ppv = precision  # Positive predictive value (same as precision)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc': auc,
        'npv': npv,
        'ppv': ppv,
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics


def evaluate_model(model, test_loader, device, save_dir=None):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save evaluation results
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(y_true, y_pred, y_pred_proba)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"NPV:         {metrics['npv']:.4f}")
    print(f"PPV:         {metrics['ppv']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {metrics['tn']}, FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}, TP: {metrics['tp']}")
    
    # Print classification report
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, 
                               target_names=['Interictal', 'Preictal'],
                               digits=4))
    
    # Save visualizations if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], 
                            save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        
        # Plot ROC curve
        plot_roc_curve(y_true, y_pred_proba,
                      save_path=os.path.join(save_dir, 'roc_curve.png'))
        
        # Plot prediction distribution
        plot_prediction_distribution(y_true, y_pred_proba,
                                    save_path=os.path.join(save_dir, 'prediction_distribution.png'))
    
    return metrics, y_true, y_pred, y_pred_proba


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Interictal', 'Preictal'],
                yticklabels=['Interictal', 'Preictal'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_true, y_pred_proba, save_path=None):
    """Plot distribution of prediction probabilities"""
    interictal_probs = y_pred_proba[y_true == 0]
    preictal_probs = y_pred_proba[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(interictal_probs, bins=50, alpha=0.5, label='Interictal', color='blue')
    plt.hist(preictal_probs, bins=50, alpha=0.5, label='Preictal', color='red')
    plt.xlabel('Predicted Probability (Preictal)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction distribution to {save_path}")
    
    plt.close()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model arguments from checkpoint
    args = checkpoint.get('args', {})
    
    # Create model
    model = create_bsdcnn_model(
        model_type=args.get('model_type', 'full'),
        num_channels=args.get('num_channels', 21),
        sequence_length=args.get('sequence_length', 5120),
        num_classes=2,
        use_binary_activation=args.get('use_binary_activation', True)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    return model, args


def main():
    parser = argparse.ArgumentParser(description='Evaluate BSDCNN for seizure prediction')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--test_patient', type=str, default=None,
                       help='Patient ID for testing')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*50)
    print("Loading model...")
    print("="*50)
    
    model, model_args = load_model_from_checkpoint(args.checkpoint_path, device)
    
    # Create data loaders
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    _, _, test_loader = create_bsdcnn_dataloaders(
        data_root=args.data_root,
        test_patient=args.test_patient or model_args.get('test_patient'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=model_args.get('window_seconds', 10),
        overlap_seconds=model_args.get('overlap_seconds', 5)
    )
    
    # Evaluate model
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50)
    
    metrics, y_true, y_pred, y_pred_proba = evaluate_model(
        model, test_loader, device, save_dir=args.output_dir
    )
    
    # Save results
    results = {
        'model_checkpoint': args.checkpoint_path,
        'test_patient': args.test_patient or model_args.get('test_patient'),
        'metrics': metrics,
        'model_args': model_args
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {results_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'predictions.npz')
        np.savez(predictions_path, 
                 y_true=y_true, 
                 y_pred=y_pred, 
                 y_pred_proba=y_pred_proba)
        print(f"Saved predictions to {predictions_path}")
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == "__main__":
    main()
