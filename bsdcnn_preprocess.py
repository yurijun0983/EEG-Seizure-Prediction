"""
Preprocess EEG data and save to disk for BSDCNN training
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import gc

from bsdcnn_data_loader import load_patient_data


# Standard 29 EEG channels available in all Siena patients
STANDARD_29_CHANNELS = [
    'EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1',
    'EEG F7', 'EEG T3', 'EEG T5',
    'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5',
    'EEG F9', 'EEG Fz', 'EEG Pz',
    'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2',
    'EEG F8', 'EEG T4', 'EEG T6',
    'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6',
    'EEG F10'
]


def preprocess_and_save_patient(data_root, patient_id, output_dir, 
                                  window_seconds=5, overlap_seconds=2.5):  # Changed: 5s window, 50% overlap
    """
    Preprocess one patient's data and save to file
    
    Args:
        data_root: Root directory of dataset
        patient_id: Patient ID (e.g., 'PN00')
        output_dir: Output directory for preprocessed data
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
    """
    print(f"\nProcessing {patient_id}...")
    
    # Load and process patient data with 29 standard channels
    preictal, interictal = load_patient_data(
        data_root, patient_id,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        target_channels=STANDARD_29_CHANNELS
    )
    
    # Convert to numpy arrays with float32 to save space
    preictal_data = np.array([seg for seg in preictal], dtype=np.float32)
    interictal_data = np.array([seg for seg in interictal], dtype=np.float32)
    
    # Free original lists
    del preictal, interictal
    gc.collect()
    
    # Save to file
    output_file = os.path.join(output_dir, f'{patient_id}.npz')
    np.savez_compressed(
        output_file,
        preictal=preictal_data,
        interictal=interictal_data
    )
    
    print(f"Saved {patient_id}: {len(preictal_data)} preictal, {len(interictal_data)} interictal")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Free memory
    del preictal_data, interictal_data
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Preprocess BSDCNN data')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--patient_ids', type=str, nargs='+', default=None,
                       help='Patient IDs to process (default: all)')
    parser.add_argument('--window_seconds', type=float, default=5,
                       help='Window size in seconds (default: 5s per papers)')
    parser.add_argument('--overlap_seconds', type=float, default=2.5,
                       help='Overlap in seconds (default: 2.5s = 50%% overlap)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get patient IDs
    if args.patient_ids is None:
        patient_ids = sorted([d for d in os.listdir(args.data_root) 
                            if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
    else:
        patient_ids = args.patient_ids
    
    print(f"Processing {len(patient_ids)} patients...")
    print(f"Patients: {patient_ids}")
    
    # Process each patient
    for patient_id in patient_ids:
        try:
            preprocess_and_save_patient(
                args.data_root, patient_id, args.output_dir,
                args.window_seconds, args.overlap_seconds
            )
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
