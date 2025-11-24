"""
Training script for BSDCNN (Binary Single-Dimensional CNN) for Seizure Prediction
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from bsdcnn_model import BSDCNN, BSDCNNSimplified, create_bsdcnn_model
from bsdcnn_data_loader import create_bsdcnn_dataloaders
from focal_loss import FocalLoss, CombinedLoss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': 0.0,
        'auc': 0.0
    }
    
    # Calculate specificity
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    
    # Calculate AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_data, batch_labels in pbar:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(val_loader, desc='Validating'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Statistics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics


def train_model(args):
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tensorboard
    log_dir = os.path.join(args.output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with augmentation for training
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    train_loader, val_loader, test_loader = create_bsdcnn_dataloaders(
        data_root=args.data_root,
        test_patient=args.test_patient,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        val_split=args.val_split,
        preprocessed_dir=args.preprocessed_dir,  # Add this parameter
        use_augmentation=True  # Enable data augmentation for training
    )
    
    # Get input dimensions from first batch
    sample_batch, _ = next(iter(train_loader))
    num_channels = sample_batch.shape[1]
    sequence_length = sample_batch.shape[2]
    
    print(f"\nInput shape: (batch_size, {num_channels}, {sequence_length})")
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    # Prepare model kwargs based on model type
    model_kwargs = {
        'num_channels': num_channels,
        'sequence_length': sequence_length,
        'num_classes': 2,
    }
    
    # Only add use_binary_activation for binary models
    if args.model_type in ['full', 'simplified']:
        model_kwargs['use_binary_activation'] = args.use_binary_activation
    
    model = create_bsdcnn_model(
        model_type=args.model_type,
        **model_kwargs
    )
    
    model = model.to(device)
    
    print(f"Model type: {args.model_type}")
    if hasattr(model, 'count_parameters'):
        print(f"Total parameters: {model.count_parameters():,}")
        print(f"Model size: {model.get_model_size():.2f} MB")
    
    # Loss function and optimizer
    if args.use_focal_loss:
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.use_weighted_loss:
        # Calculate class weights
        n_train = len(train_loader.dataset)
        n_positive = sum([label.item() for _, label in train_loader.dataset])
        n_negative = n_train - n_positive
        
        weight = torch.FloatTensor([1.0, n_negative / n_positive]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        print(f"Using weighted loss with weights: {weight.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # All GA-related code has been removed
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_f1 = val_metrics['f1']
            
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': best_val_f1,
                'args': vars(args)
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'args': vars(args),
        'training_history': training_history,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {results_path}")
    
    writer.close()
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train BSDCNN for seizure prediction')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files (optional)')
    parser.add_argument('--test_patient', type=str, default=None,
                       help='Patient ID for testing (leave-one-out)')
    parser.add_argument('--window_seconds', type=float, default=5,
                       help='Window size in seconds (default: 5s per papers)')
    parser.add_argument('--overlap_seconds', type=float, default=2.5,
                       help='Overlap in seconds (default: 2.5s = 50%% overlap)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='full',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    parser.add_argument('--use_binary_activation', action='store_true',
                       help='Use binary activation functions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--use_weighted_loss', action='store_true',
                       help='Use weighted loss for class imbalance')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use Focal Loss instead of weighted CE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal Loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma parameter')
    parser.add_argument('--strong_augmentation', action='store_true',
                       help='Use stronger augmentation for better cross-patient generalization')
    
    # GA training parameters (removed)
    # NSGA-II Patient Selection parameters (removed)
    
    # System parameters
    parser.add_argument('--output_dir', type=str, default='outputs_bsdcnn',
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("Configuration")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    # All GA-related optimization code has been removed
    # Use the new domain adaptation pipeline instead: run_complete_pipeline.py
    
    # ========== Standard Training (原有流程) ==========
    # Train model
    model, results = train_model(args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()
