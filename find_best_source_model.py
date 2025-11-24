"""
Step 2: Find the best source model (anchor) for the test patient
Test all individual patient models on the test patient to find which performs best
"""

import os
import argparse
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import load_patient_data, BSDCNNDataset, STANDARD_29_CHANNELS
from torch.utils.data import DataLoader


def evaluate_model_on_patient(
    model,
    test_patient_id,
    data_root,
    preprocessed_dir,
    batch_size=256,
    window_seconds=5,
    overlap_seconds=2.5,
    device='cuda'
):
    """
    Evaluate a trained model on a test patient
    
    Args:
        model: Trained PyTorch model
        test_patient_id: Test patient ID
        data_root: Root directory of dataset
        preprocessed_dir: Directory with preprocessed .npz files
        batch_size: Batch size
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
        device: Device to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load test patient data
    if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{test_patient_id}.npz')):
        data_file = np.load(os.path.join(preprocessed_dir, f'{test_patient_id}.npz'))
        preictal = list(data_file['preictal'])
        interictal = list(data_file['interictal'])
        data_file.close()
    else:
        preictal, interictal = load_patient_data(
            data_root, test_patient_id,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            target_channels=STANDARD_29_CHANNELS
        )
    
    # Convert to numpy arrays
    preictal = [np.array(seg, dtype=np.float32) for seg in preictal]
    interictal = [np.array(seg, dtype=np.float32) for seg in interictal]
    
    # Create dataset
    all_segments = preictal + interictal
    all_labels = [1] * len(preictal) + [0] * len(interictal)
    
    test_dataset = BSDCNNDataset(all_segments, all_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels_list = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc=f'Testing on {test_patient_id}'):
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels_list.extend(batch_labels.numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels_list, all_preds)),
        'precision': float(precision_score(all_labels_list, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels_list, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels_list, all_preds, zero_division=0)),
        'n_samples': int(len(all_labels_list)),
        'n_preictal': int(sum(all_labels_list)),
        'n_interictal': int(len(all_labels_list) - sum(all_labels_list))
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Find best source model for test patient')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files')
    parser.add_argument('--models_dir', type=str, default='outputs_individual_models',
                       help='Directory containing individual patient models')
    parser.add_argument('--test_patient', type=str, required=True,
                       help='Test patient ID (e.g., PN14)')
    parser.add_argument('--output_file', type=str, default='best_source_model.json',
                       help='Output JSON file with results')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    parser.add_argument('--window_seconds', type=float, default=5,
                       help='Window size in seconds')
    parser.add_argument('--overlap_seconds', type=float, default=2.5,
                       help='Overlap in seconds')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all patient IDs except test patient
    all_patients = sorted([d for d in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
    source_patients = [p for p in all_patients if p != args.test_patient]
    
    print(f"\nTest patient: {args.test_patient}")
    print(f"Source patients: {source_patients}\n")
    
    # Load a sample to get dimensions
    if args.preprocessed_dir:
        sample_file = os.path.join(args.preprocessed_dir, f'{source_patients[0]}.npz')
        data = np.load(sample_file)
        sample_seg = data['preictal'][0]
        data.close()
    else:
        preictal, _ = load_patient_data(
            args.data_root, source_patients[0],
            window_seconds=args.window_seconds,
            overlap_seconds=args.overlap_seconds,
            target_channels=STANDARD_29_CHANNELS
        )
        sample_seg = preictal[0]
    
    num_channels = sample_seg.shape[0]
    sequence_length = sample_seg.shape[1]
    
    print(f"Input shape: ({num_channels}, {sequence_length})\n")
    
    # Test each source model on test patient
    results = {}
    
    for source_patient in source_patients:
        print(f"\n{'='*60}")
        print(f"Testing {source_patient} model on {args.test_patient}")
        print(f"{'='*60}")
        
        model_path = os.path.join(args.models_dir, f'{source_patient}_model.pth')
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found for {source_patient}, skipping...")
            continue
        
        # Load model
        model = create_bsdcnn_model(
            model_type=args.model_type,
            num_channels=num_channels,
            sequence_length=sequence_length,
            num_classes=2
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        metrics = evaluate_model_on_patient(
            model=model,
            test_patient_id=args.test_patient,
            data_root=args.data_root,
            preprocessed_dir=args.preprocessed_dir,
            batch_size=args.batch_size,
            window_seconds=args.window_seconds,
            overlap_seconds=args.overlap_seconds,
            device=device
        )
        
        results[source_patient] = metrics
        
        print(f"\nResults for {source_patient} on {args.test_patient}:")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Find best source model
    if results:
        best_source = max(results.items(), key=lambda x: x[1]['f1'])
        best_patient = best_source[0]
        best_metrics = best_source[1]
        
        print(f"\n{'='*60}")
        print("BEST SOURCE MODEL (ANCHOR)")
        print(f"{'='*60}")
        print(f"Anchor patient: {best_patient}")
        print(f"F1 Score on {args.test_patient}: {best_metrics['f1']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        
        # Save results
        output = {
            'test_patient': args.test_patient,
            'anchor_patient': best_patient,
            'anchor_metrics': best_metrics,
            'all_results': results,
            'args': vars(args)
        }
        
        output_path = os.path.join(args.models_dir, args.output_file)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nERROR: No valid results obtained!")


if __name__ == '__main__':
    main()
