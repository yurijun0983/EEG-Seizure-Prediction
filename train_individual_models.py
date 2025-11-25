"""
Step 1: Train individual models for each patient
Each patient gets their own dedicated seizure prediction model
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import load_patient_data, BSDCNNDataset, STANDARD_29_CHANNELS
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def train_single_patient_model(
    patient_id,
    data_root,
    preprocessed_dir,
    output_dir,
    epochs=50,
    batch_size=256,
    lr=0.001,
    model_type='standard',
    window_seconds=5,
    overlap_seconds=2.5
):
    """
    Train a model for a single patient using only their data
    
    Args:
        patient_id: Patient ID (e.g., 'PN00')
        data_root: Root directory of dataset
        preprocessed_dir: Directory with preprocessed .npz files
        output_dir: Output directory for models
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        model_type: Model architecture type
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
    
    Returns:
        Trained model and training metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for {patient_id}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load patient data
    if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{patient_id}.npz')):
        print(f"Loading preprocessed data for {patient_id}...")
        data_file = np.load(os.path.join(preprocessed_dir, f'{patient_id}.npz'))
        preictal = list(data_file['preictal'])
        interictal = list(data_file['interictal'])
        data_file.close()
    else:
        print(f"Processing raw data for {patient_id}...")
        preictal, interictal = load_patient_data(
            data_root, patient_id,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            target_channels=STANDARD_29_CHANNELS
        )
    
    # Convert to numpy arrays
    preictal = [np.array(seg, dtype=np.float32) for seg in preictal]
    interictal = [np.array(seg, dtype=np.float32) for seg in interictal]
    
    print(f"Loaded: {len(preictal)} preictal, {len(interictal)} interictal")
    
    if len(preictal) == 0 or len(interictal) == 0:
        print(f"WARNING: Insufficient data for {patient_id}, skipping...")
        return None, None
    
    # Create dataset
    all_segments = preictal + interictal
    all_labels = [1] * len(preictal) + [0] * len(interictal)
    
    # Train/val split
    train_segs, val_segs, train_labels, val_labels = train_test_split(
        all_segments, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    train_dataset = BSDCNNDataset(train_segs, train_labels, augment=True)
    val_dataset = BSDCNNDataset(val_segs, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get input dimensions
    sample_seg = all_segments[0]
    num_channels = sample_seg.shape[0]
    sequence_length = sample_seg.shape[1]
    
    # Create model
    model = create_bsdcnn_model(
        model_type=model_type,
        num_channels=num_channels,
        sequence_length=sequence_length,
        num_classes=2
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels_epoch = []
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_epoch.extend(batch_labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels_epoch, train_preds, zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_epoch = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_epoch.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels_epoch, val_preds, zero_division=0)
        val_acc = accuracy_score(val_labels_epoch, val_preds)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, F1={train_f1:.4f} | "
              f"Val Loss={val_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc
        })
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            model_save_path = os.path.join(output_dir, f'{patient_id}_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'patient_id': patient_id
            }, model_save_path)
    
    print(f"\nBest model at epoch {best_epoch} with Val F1={best_val_f1:.4f}")
    
    # Save training history
    history_path = os.path.join(output_dir, f'{patient_id}_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'patient_id': patient_id,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'history': history
        }, f, indent=4)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train individual patient models')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, default='outputs_individual_models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per patient')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    parser.add_argument('--window_seconds', type=float, default=5,
                       help='Window size in seconds')
    parser.add_argument('--overlap_seconds', type=float, default=2.5,
                       help='Overlap in seconds')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all patient IDs
    all_patients = sorted([d for d in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
    
    print(f"\nTraining individual models for {len(all_patients)} patients")
    print(f"Patients: {all_patients}\n")
    
    # Train each patient
    results = {}
    for patient_id in all_patients:
        try:
            model, history = train_single_patient_model(
                patient_id=patient_id,
                data_root=args.data_root,
                preprocessed_dir=args.preprocessed_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                model_type=args.model_type,
                window_seconds=args.window_seconds,
                overlap_seconds=args.overlap_seconds
            )
            
            if history:
                best_val_f1 = max([h['val_f1'] for h in history])
                results[patient_id] = best_val_f1
            
        except Exception as e:
            print(f"ERROR training {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'patients': all_patients,
            'results': results,
            'args': vars(args)
        }, f, indent=4)
    
    print(f"\n{'='*60}")
    print("All individual models trained!")
    print(f"{'='*60}")
    print("\nBest validation F1 scores:")
    for patient_id, f1 in results.items():
        print(f"  {patient_id}: {f1:.4f}")


if __name__ == '__main__':
    main()
