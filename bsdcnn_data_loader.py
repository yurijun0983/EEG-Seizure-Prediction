"""
Data loader for BSDCNN model using Siena Scalp EEG Database
Processes raw EEG signals for seizure prediction
"""

import os
import numpy as np
import pyedflib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


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


class EEGAugmentation:
    """Enhanced EEG-specific data augmentation for better cross-patient generalization"""
    
    @staticmethod
    def add_gaussian_noise(x, std=0.01):
        """Add Gaussian noise to simulate sensor noise"""
        noise = np.random.normal(0, std, x.shape)
        return x + noise
    
    @staticmethod
    def time_shift(x, max_shift=50):
        """Time shift to simulate temporal variations"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(x, shift, axis=1)
    
    @staticmethod
    def amplitude_scale(x, scale_range=(0.95, 1.05)):
        """Amplitude scaling to simulate inter-patient amplitude differences"""
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    @staticmethod
    def channel_dropout(x, dropout_prob=0.1):
        """Randomly drop channels to improve robustness"""
        mask = np.random.binomial(1, 1 - dropout_prob, size=x.shape[0])
        mask = mask.reshape(-1, 1)  # (channels, 1)
        return x * mask
    
    @staticmethod
    def frequency_shift(x, max_shift_hz=2, sampling_rate=512):
        """Slight frequency shift using FFT (simulates inter-patient frequency variations)"""
        # Apply FFT
        x_fft = np.fft.rfft(x, axis=1)
        freqs = np.fft.rfftfreq(x.shape[1], 1/sampling_rate)
        
        # Random shift in Hz
        shift_hz = np.random.uniform(-max_shift_hz, max_shift_hz)
        shift_bins = int(shift_hz * x.shape[1] / sampling_rate)
        
        # Roll frequency bins
        if shift_bins != 0:
            x_fft = np.roll(x_fft, shift_bins, axis=1)
        
        # Inverse FFT
        x_shifted = np.fft.irfft(x_fft, n=x.shape[1], axis=1)
        return x_shifted.real
    
    @staticmethod
    def mixup(x1, x2, alpha=0.2):
        """Mixup augmentation: combine two samples"""
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2
    
    @staticmethod
    def random_window_warp(x, warp_range=0.1):
        """Randomly warp time windows (simulates temporal distortions)"""
        num_samples = x.shape[1]
        warp_factor = np.random.uniform(1 - warp_range, 1 + warp_range)
        
        # Create warped indices
        old_indices = np.arange(num_samples)
        new_length = int(num_samples * warp_factor)
        new_indices = np.linspace(0, num_samples - 1, new_length)
        
        # Interpolate
        x_warped = np.zeros((x.shape[0], num_samples))
        for ch in range(x.shape[0]):
            x_warped[ch] = np.interp(old_indices, new_indices, 
                                     np.interp(new_indices, old_indices, x[ch]))
        return x_warped
    
    def __call__(self, x, augment_prob=0.5, strong_aug=False):
        """Apply random augmentations
        
        Args:
            x: Input signal (channels, samples)
            augment_prob: Probability of applying each augmentation
            strong_aug: If True, use more aggressive augmentation for better generalization
        """
        x_aug = x.copy()
        
        # Standard augmentations (always available)
        if np.random.rand() < augment_prob:
            x_aug = self.add_gaussian_noise(x_aug, std=0.01 if not strong_aug else 0.02)
        if np.random.rand() < augment_prob:
            x_aug = self.time_shift(x_aug, max_shift=50 if not strong_aug else 100)
        if np.random.rand() < augment_prob:
            x_aug = self.amplitude_scale(x_aug, scale_range=(0.95, 1.05) if not strong_aug else (0.9, 1.1))
        
        # Enhanced augmentations for cross-patient generalization
        if strong_aug:
            if np.random.rand() < augment_prob * 0.3:  # Less frequent
                x_aug = self.channel_dropout(x_aug, dropout_prob=0.1)
            if np.random.rand() < augment_prob * 0.2:
                x_aug = self.frequency_shift(x_aug, max_shift_hz=2)
            if np.random.rand() < augment_prob * 0.2:
                x_aug = self.random_window_warp(x_aug, warp_range=0.1)
        
        return x_aug


class BSDCNNDataset(Dataset):
    """
    Dataset for BSDCNN that loads raw EEG segments
    
    Args:
        segments: List of EEG segments (numpy arrays)
        labels: List of labels (0: interictal, 1: preictal)
        transform: Optional transform to apply to segments
        augment: Whether to apply data augmentation (for training only)
        strong_augment: Whether to use stronger augmentation for cross-patient generalization
    """
    def __init__(self, segments, labels, transform=None, augment=False, strong_augment=False):
        self.segments = segments
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.strong_augment = strong_augment
        if augment:
            self.augmentation = EEGAugmentation()
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        
        # Apply augmentation for preictal samples (label=1) during training
        if self.augment and label == 1 and np.random.rand() < 0.5:
            segment = self.augmentation(segment, augment_prob=0.5, strong_aug=self.strong_augment)
        
        # Convert to tensor
        segment = torch.FloatTensor(segment)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            segment = self.transform(segment)
        
        return segment, label


class EEGPreprocessor:
    """
    Preprocessing pipeline for EEG signals
    """
    def __init__(self, sampling_rate=512, lowcut=0.5, highcut=70.0, notch_freq=50.0):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
    
    def bandpass_filter(self, data):
        """Apply bandpass filter"""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data
    
    def notch_filter(self, data):
        """Apply notch filter to remove power line interference"""
        nyquist = 0.5 * self.sampling_rate
        freq = self.notch_freq / nyquist
        b, a = signal.iirnotch(freq, Q=30.0, fs=self.sampling_rate)
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data
    
    def normalize(self, data):
        """Z-score normalization per channel"""
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        normalized_data = (data - mean) / std
        return normalized_data
    
    def preprocess(self, data):
        """Apply full preprocessing pipeline"""
        data = self.bandpass_filter(data)
        data = self.notch_filter(data)
        # DO NOT normalize here - will normalize per-segment later if needed
        # Normalization at this level removes all discriminative information
        return data


def load_edf_file(file_path, target_channels=None):
    """
    Load EEG data from EDF file
    
    Args:
        file_path: Path to EDF file
        target_channels: List of channel names to load (None = load all)
    
    Returns:
        data: EEG data (channels, samples)
        channel_names: List of channel names
        sampling_rate: Sampling rate in Hz
    """
    try:
        f = pyedflib.EdfReader(file_path)
        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        sampling_rate = f.getSampleFrequency(0)
        
        # Load data
        if target_channels is None:
            # Load all channels
            data = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                data[i, :] = f.readSignal(i)
        else:
            # Load specific channels
            selected_indices = []
            selected_names = []
            for i, name in enumerate(channel_names):
                if name in target_channels:
                    selected_indices.append(i)
                    selected_names.append(name)
            
            if len(selected_indices) == 0:
                f.close()
                return None, None, None
            
            data = np.zeros((len(selected_indices), f.getNSamples()[selected_indices[0]]))
            for i, idx in enumerate(selected_indices):
                data[i, :] = f.readSignal(idx)
            
            channel_names = selected_names
        
        f.close()
        return data, channel_names, sampling_rate
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None


def parse_seizure_list(seizure_file):
    """
    Parse seizure list file
    
    Args:
        seizure_file: Path to Seizures-list-*.txt file
    
    Returns:
        List of (file_name, start_time, end_time) tuples (times in seconds from file start)
    """
    seizures = []
    
    if not os.path.exists(seizure_file):
        return seizures
    
    with open(seizure_file, 'r') as f:
        lines = f.readlines()
    
    # Parse seizure entries
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "Seizure n X" pattern (with optional notes like "(in sleep)" and optional ":")
        if 'Seizure n' in line:
            file_name = None
            reg_start_time = None
            start_time = None
            end_time = None
            
            # Parse the next few lines for this seizure
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                
                # File name
                if next_line.startswith('File name:'):
                    file_name = next_line.split(':', 1)[1].strip()
                
                # Registration start time
                elif 'Registration start time' in next_line:
                    # Extract time after 'time' keyword, handling both 'time:' and 'time :'
                    if 'time:' in next_line:
                        time_str = next_line.split('time:', 1)[1].strip()
                    elif 'time :' in next_line:
                        time_str = next_line.split('time :', 1)[1].strip()
                    else:
                        continue
                    reg_start_time = parse_time_string(time_str)
                
                # Seizure start time
                elif next_line.startswith('start time'):
                    if 'time:' in next_line:
                        time_str = next_line.split('time:', 1)[1].strip()
                    elif 'time :' in next_line:
                        time_str = next_line.split('time :', 1)[1].strip()
                    else:
                        continue
                    start_time = parse_time_string(time_str)
                
                # Seizure end time
                elif next_line.startswith('end time'):
                    if 'time:' in next_line:
                        time_str = next_line.split('time:', 1)[1].strip()
                    elif 'time :' in next_line:
                        time_str = next_line.split('time :', 1)[1].strip()
                    else:
                        continue
                    end_time = parse_time_string(time_str)
                    break  # Found all info for this seizure
            
            # Calculate times relative to recording start
            if all([file_name, reg_start_time is not None, 
                   start_time is not None, end_time is not None]):
                # Convert times to seconds from recording start
                start_offset = calculate_time_offset(reg_start_time, start_time)
                end_offset = calculate_time_offset(reg_start_time, end_time)
                
                seizures.append((file_name, start_offset, end_offset))
        
        i += 1
    
    return seizures


def parse_time_string(time_str):
    """
    Parse time string in HH:MM:SS format to total seconds
    
    Args:
        time_str: Time string like "19:58:36"
    
    Returns:
        Total seconds as integer, or None if parsing fails
    """
    try:
        time_str = time_str.strip()
        # Use colon as separator (format is now unified)
        parts = time_str.split(':')
        
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        print(f"Warning: Failed to parse time string '{time_str}': {e}")
        return None
    
    return None


def calculate_time_offset(reg_start, event_time):
    """
    Calculate offset in seconds from registration start to event time
    
    Args:
        reg_start: Registration start time in seconds (from midnight)
        event_time: Event time in seconds (from midnight)
    
    Returns:
        Offset in seconds from registration start
    """
    offset = event_time - reg_start
    
    # Handle day boundary crossing (if event is next day)
    if offset < 0:
        offset += 24 * 3600  # Add 24 hours
    
    return offset


def create_segments(data, window_size, overlap):
    """
    Create overlapping segments from continuous EEG data
    
    Args:
        data: EEG data (channels, samples)
        window_size: Window size in samples
        overlap: Overlap in samples
    
    Returns:
        segments: List of segments (channels, window_size)
    """
    n_channels, n_samples = data.shape
    step = window_size - overlap
    
    segments = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        segment = data[:, start:end]
        segments.append(segment)
    
    return segments


def load_patient_data(data_root, patient_id, 
                      window_seconds=5,  # Changed from 10 to 5 (paper standard)
                      overlap_seconds=2.5,  # 50% overlap (paper standard)
                      preictal_duration=15*60,  # 15 minutes (paper standard, not 30)
                      preictal_offset=0,  # 0 minutes - immediately before seizure
                      postictal_duration=5*60,  # 5 minutes after seizure
                      interictal_gap=4*60*60,  # 4 hours minimum gap (NEW - paper requirement)
                      target_channels=None,
                      max_interictal_ratio=5,
                      max_segments_per_file=500):
    """
    Load data for a single patient
    
    Args:
        data_root: Root directory of the dataset
        patient_id: Patient ID (e.g., 'PN00')
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
        preictal_duration: Duration of preictal period (seconds before seizure)
        preictal_offset: Offset before seizure start (seconds)
        postictal_duration: Duration to exclude after seizure (seconds)
        target_channels: List of channel names to use
        max_interictal_ratio: Maximum ratio of interictal to preictal samples
    
    Returns:
        preictal_segments: List of preictal segments
        interictal_segments: List of interictal segments
    """
    # Use standard 29 channels if not specified
    if target_channels is None:
        target_channels = STANDARD_29_CHANNELS
    
    patient_dir = os.path.join(data_root, patient_id)
    seizure_file = os.path.join(patient_dir, f'Seizures-list-{patient_id}.txt')
    
    # Parse seizure information
    seizures = parse_seizure_list(seizure_file)
    print(f"Found {len(seizures)} seizures for {patient_id}")
    
    # Load all EDF files
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])
    
    preictal_segments = []
    interictal_segments = []
    
    preprocessor = EEGPreprocessor()
    
    for edf_file in tqdm(edf_files, desc=f"Processing {patient_id}"):
        file_path = os.path.join(patient_dir, edf_file)
        
        # Load EEG data
        data, channel_names, sampling_rate = load_edf_file(file_path, target_channels)
        
        if data is None:
            continue
        
        # Preprocess
        data = preprocessor.preprocess(data)
        
        # Convert time to samples
        window_size = int(window_seconds * sampling_rate)
        overlap = int(overlap_seconds * sampling_rate)
        
        # Find seizures in this file
        file_seizures = [(s, e) for f, s, e in seizures if f == edf_file]
        
        # Mark time periods
        n_samples = data.shape[1]
        is_preictal = np.zeros(n_samples, dtype=bool)
        is_excluded = np.zeros(n_samples, dtype=bool)
        
        for seizure_start, seizure_end in file_seizures:
            # Convert to samples
            seizure_start_sample = int(seizure_start * sampling_rate)
            seizure_end_sample = int(seizure_end * sampling_rate)
            
            # Mark preictal period (15 minutes IMMEDIATELY before seizure - paper standard)
            ideal_preictal_start = seizure_start_sample - int(preictal_duration * sampling_rate)
            ideal_preictal_end = seizure_start_sample  # No offset - up to seizure onset
            
            # Adjust if preictal window starts before recording
            preictal_start = max(0, ideal_preictal_start)
            preictal_end = ideal_preictal_end
            
            # Only mark preictal if we have at least some valid window
            if preictal_end > 0 and preictal_end > preictal_start:
                is_preictal[preictal_start:preictal_end] = True
                print(f"  Seizure at {seizure_start}s: preictal window [{preictal_start/sampling_rate:.1f}s - {preictal_end/sampling_rate:.1f}s] ({(preictal_end-preictal_start)/sampling_rate/60:.1f} min)")
            
            # Mark excluded periods for interictal selection
            # Safety margin: 30 min before preictal starts
            # Seizure (ictal) and postictal periods
            # NOTE: Preictal itself is NOT excluded - we want to keep those segments!
            
            safety_margin = 30*60  # 30 minutes in seconds
            
            # Before preictal: safety margin
            exclusion_before = max(0, ideal_preictal_start - int(safety_margin * sampling_rate))
            is_excluded[exclusion_before:preictal_start] = True
            
            # After preictal: seizure (ictal) + postictal
            postictal_end = seizure_end_sample + int(postictal_duration * sampling_rate)
            exclusion_after = min(n_samples, postictal_end)
            is_excluded[seizure_start_sample:exclusion_after] = True
        
        # Create segments with limits
        step = window_size - overlap
        file_segment_count = 0
        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            
            # Check segment label
            segment_samples = np.arange(start, end)
            
            # Skip if any part is in excluded zone
            if np.any(is_excluded[segment_samples]):
                continue
            
            # Extract segment as float32 immediately to reduce memory
            segment = data[:, start:end].astype(np.float32)
            
            # Check if majority is preictal
            if np.mean(is_preictal[segment_samples]) > 0.5:
                preictal_segments.append(segment)
            else:
                # Interictal - apply sampling to reduce memory
                if file_segment_count < max_segments_per_file or len(file_seizures) > 0:
                    interictal_segments.append(segment)
                    file_segment_count += 1
        
        # Delete raw data immediately after processing this file
        del data, is_preictal, is_excluded
        import gc
        gc.collect()
    
    # Balance dataset using 1:1 sampling (based on latest research)
    # Randomly undersample interictal to match preictal count
    if len(preictal_segments) > 0 and len(interictal_segments) > len(preictal_segments):
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(interictal_segments), 
                                  size=len(preictal_segments),  # 1:1 ratio
                                  replace=False)
        interictal_segments = [interictal_segments[i] for i in indices]
        print(f"  Balanced sampling: {len(preictal_segments)} preictal + {len(interictal_segments)} interictal (1:1 ratio)")
    
    print(f"{patient_id}: {len(preictal_segments)} preictal, {len(interictal_segments)} interictal segments")
    
    return preictal_segments, interictal_segments


def create_bsdcnn_dataloaders(data_root, 
                               patient_ids=None,
                               test_patient=None,
                               batch_size=32,
                               num_workers=4,
                               window_seconds=10,
                               overlap_seconds=5,
                               target_channels=None,
                               val_split=0.2,
                               preprocessed_dir=None,
                               use_augmentation=False):  # New parameter
    """
    Create train/val/test dataloaders for BSDCNN
    
    Args:
        data_root: Root directory of dataset
        patient_ids: List of patient IDs to include (None = all)
        test_patient: Patient ID for testing (leave-one-out)
        batch_size: Batch size
        num_workers: Number of data loading workers
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
        target_channels: List of channel names to use
        val_split: Validation split ratio
        preprocessed_dir: Directory with preprocessed .npz files (optional)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get all patient IDs if not specified
    if patient_ids is None:
        patient_ids = sorted([d for d in os.listdir(data_root) 
                            if os.path.isdir(os.path.join(data_root, d)) and d.startswith('PN')])
    
    # Split patients for training and testing
    if test_patient:
        train_patients = [p for p in patient_ids if p != test_patient]
        test_patients = [test_patient]
    else:
        # Random split
        train_patients = patient_ids[:int(0.8 * len(patient_ids))]
        test_patients = patient_ids[int(0.8 * len(patient_ids)):]
    
    print(f"Training patients: {train_patients}")
    print(f"Test patients: {test_patients}")
    
    # Load training data
    train_preictal = []
    train_interictal = []
    
    for patient_id in train_patients:
        # Try to load from preprocessed file first
        if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{patient_id}.npz')):
            print(f"Loading preprocessed data for {patient_id}...")
            data_file = np.load(os.path.join(preprocessed_dir, f'{patient_id}.npz'))
            preictal = list(data_file['preictal'])
            interictal = list(data_file['interictal'])
            data_file.close()
        else:
            # Process from raw data
            preictal, interictal = load_patient_data(
                data_root, patient_id, 
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                target_channels=target_channels
            )
        
        # Immediately convert to numpy arrays and extend
        train_preictal.extend([np.array(seg, dtype=np.float32) for seg in preictal])
        train_interictal.extend([np.array(seg, dtype=np.float32) for seg in interictal])
        
        # Force garbage collection after each patient
        del preictal, interictal
        import gc
        gc.collect()
    
    # Create train/val split IMMEDIATELY to free memory
    print(f"\nCreating training dataset...")
    train_segments = train_preictal + train_interictal
    train_labels = [1] * len(train_preictal) + [0] * len(train_interictal)
    
    # Free the original lists
    del train_preictal, train_interictal
    import gc
    gc.collect()
    
    train_segments, val_segments, train_labels, val_labels = train_test_split(
        train_segments, train_labels, test_size=val_split, 
        stratify=train_labels, random_state=42
    )
    
    # Create datasets immediately and free segment lists
    train_dataset = BSDCNNDataset(train_segments, train_labels, augment=use_augmentation)
    val_dataset = BSDCNNDataset(val_segments, val_labels, augment=False)
    
    del train_segments, val_segments, train_labels, val_labels
    gc.collect()
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Load test data
    test_preictal = []
    test_interictal = []
    
    for patient_id in test_patients:
        # Try to load from preprocessed file first
        if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{patient_id}.npz')):
            print(f"Loading preprocessed data for {patient_id}...")
            data_file = np.load(os.path.join(preprocessed_dir, f'{patient_id}.npz'))
            preictal = list(data_file['preictal'])
            interictal = list(data_file['interictal'])
            data_file.close()
        else:
            # Process from raw data
            preictal, interictal = load_patient_data(
                data_root, patient_id,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                target_channels=target_channels
            )
        
        # Immediately convert to numpy arrays and extend
        test_preictal.extend([np.array(seg, dtype=np.float32) for seg in preictal])
        test_interictal.extend([np.array(seg, dtype=np.float32) for seg in interictal])
        
        # Force garbage collection after each patient
        del preictal, interictal
        gc.collect()
    
    # Create test set and free memory
    print(f"\nCreating test dataset...")
    test_segments = test_preictal + test_interictal
    test_labels = [1] * len(test_preictal) + [0] * len(test_interictal)
    
    del test_preictal, test_interictal
    gc.collect()
    
    # Create datasets
    test_dataset = BSDCNNDataset(test_segments, test_labels, augment=False)
    
    del test_segments, test_labels
    gc.collect()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def load_single_patient_dataloader(
    data_root,
    patient_id,
    batch_size=32,
    num_workers=0,
    window_seconds=10,
    overlap_seconds=5,
    preprocessed_dir=None,
    target_channels=None
):
    """
    Load data for a single patient only (for threshold optimization on test set)
    
    Args:
        data_root: Root directory of the dataset
        patient_id: Patient ID to load (e.g., 'PN14')
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
        preprocessed_dir: Directory with preprocessed .npz files
        target_channels: List of target channel names
    
    Returns:
        DataLoader for the specified patient
    """
    print(f"\nLoading data for patient: {patient_id}")
    
    # Try to load from preprocessed file first
    if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{patient_id}.npz')):
        print(f"Loading preprocessed data for {patient_id}...")
        data_file = np.load(os.path.join(preprocessed_dir, f'{patient_id}.npz'))
        preictal = list(data_file['preictal'])
        interictal = list(data_file['interictal'])
        data_file.close()
    else:
        # Process from raw data
        print(f"Processing raw data for {patient_id}...")
        preictal, interictal = load_patient_data(
            data_root, patient_id,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            target_channels=target_channels
        )
    
    # Convert to numpy arrays
    preictal = [np.array(seg, dtype=np.float32) for seg in preictal]
    interictal = [np.array(seg, dtype=np.float32) for seg in interictal]
    
    # Create dataset
    segments = preictal + interictal
    labels = [1] * len(preictal) + [0] * len(interictal)
    
    print(f"Loaded {len(preictal)} preictal and {len(interictal)} interictal segments")
    
    dataset = BSDCNNDataset(segments, labels, augment=False)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Total samples: {len(dataset)}")
    
    return dataloader


def create_bsdcnn_dataloaders_with_patient_selection(
    data_root,
    selected_source_patients,
    test_patient=None,
    batch_size=32,
    num_workers=0,
    window_seconds=10,
    overlap_seconds=5,
    val_split=0.2,
    preprocessed_dir=None,
    target_channels=None,
    use_augmentation=False,
    strong_augmentation=False,
    patient_weights=None
):
    """
    Create dataloaders with patient selection (for GA optimization)
    
    Args:
        data_root: Root directory of the dataset
        selected_source_patients: List of selected source patient IDs for training
        test_patient: Patient ID for testing (leave-one-out)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        window_seconds: Window size in seconds
        overlap_seconds: Overlap in seconds
        val_split: Validation split ratio
        preprocessed_dir: Directory with preprocessed .npz files
        target_channels: List of target channel names
        use_augmentation: Whether to use data augmentation
        strong_augmentation: Whether to use stronger augmentation
        patient_weights: Optional weights for each patient (for weighted loss)
    
    Returns:
        train_loader, val_loader, test_loader, class_counts
    """
    print(f"\nLoading data with patient selection...")
    print(f"Selected source patients: {selected_source_patients}")
    print(f"Test patient: {test_patient}")
    
    # Load data from selected patients
    all_preictal = []
    all_interictal = []
    
    for patient_id in selected_source_patients:
        if patient_id == test_patient:
            continue  # Skip test patient
        
        # Try preprocessed first
        if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{patient_id}.npz')):
            data_file = np.load(os.path.join(preprocessed_dir, f'{patient_id}.npz'))
            preictal = list(data_file['preictal'])
            interictal = list(data_file['interictal'])
            data_file.close()
        else:
            # Process from raw
            preictal, interictal = load_patient_data(
                data_root, patient_id,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                target_channels=target_channels
            )
        
        # Apply patient weights if provided
        if patient_weights is not None:
            patient_idx = selected_source_patients.index(patient_id)
            weight = patient_weights[patient_idx]
            
            # Sample based on weight (simple approach)
            if weight < 1.0:
                n_preictal = int(len(preictal) * weight)
                n_interictal = int(len(interictal) * weight)
                preictal = preictal[:n_preictal] if n_preictal > 0 else preictal[:1]
                interictal = interictal[:n_interictal] if n_interictal > 0 else interictal[:1]
        
        all_preictal.extend(preictal)
        all_interictal.extend(interictal)
        
        print(f"  {patient_id}: {len(preictal)} preictal, {len(interictal)} interictal")
    
    # Convert to numpy
    all_preictal = [np.array(seg, dtype=np.float32) for seg in all_preictal]
    all_interictal = [np.array(seg, dtype=np.float32) for seg in all_interictal]
    
    print(f"\nTotal training data:")
    print(f"  Preictal: {len(all_preictal)}")
    print(f"  Interictal: {len(all_interictal)}")
    
    # Train/Val split
    if len(all_preictal) > 0 and len(all_interictal) > 0:
        train_preictal, val_preictal = train_test_split(
            all_preictal, test_size=val_split, random_state=42, shuffle=True
        )
        train_interictal, val_interictal = train_test_split(
            all_interictal, test_size=val_split, random_state=42, shuffle=True
        )
    else:
        train_preictal, val_preictal = all_preictal, []
        train_interictal, val_interictal = all_interictal, []
    
    # Create train dataset
    train_segments = train_preictal + train_interictal
    train_labels = [1] * len(train_preictal) + [0] * len(train_interictal)
    train_dataset = BSDCNNDataset(
        train_segments, train_labels,
        augment=use_augmentation,
        strong_augment=strong_augmentation
    )
    
    # Create val dataset
    val_segments = val_preictal + val_interictal
    val_labels = [1] * len(val_preictal) + [0] * len(val_interictal)
    val_dataset = BSDCNNDataset(val_segments, val_labels, augment=False)
    
    # Load test patient data
    if test_patient:
        if preprocessed_dir and os.path.exists(os.path.join(preprocessed_dir, f'{test_patient}.npz')):
            data_file = np.load(os.path.join(preprocessed_dir, f'{test_patient}.npz'))
            test_preictal = list(data_file['preictal'])
            test_interictal = list(data_file['interictal'])
            data_file.close()
        else:
            test_preictal, test_interictal = load_patient_data(
                data_root, test_patient,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                target_channels=target_channels
            )
        
        test_preictal = [np.array(seg, dtype=np.float32) for seg in test_preictal]
        test_interictal = [np.array(seg, dtype=np.float32) for seg in test_interictal]
        
        test_segments = test_preictal + test_interictal
        test_labels = [1] * len(test_preictal) + [0] * len(test_interictal)
        test_dataset = BSDCNNDataset(test_segments, test_labels, augment=False)
    else:
        test_dataset = BSDCNNDataset([], [], augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Class counts
    class_counts = [len(train_interictal), len(train_preictal)]
    
    print(f"\nDataset created:")
    print(f"  Train: {len(train_dataset)} ({len(train_preictal)} preictal, {len(train_interictal)} interictal)")
    print(f"  Val: {len(val_dataset)} ({len(val_preictal)} preictal, {len(val_interictal)} interictal)")
    print(f"  Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_counts


if __name__ == "__main__":
    # Test data loading
    data_root = "data/siena-scalp-eeg-database-1.0.0"
    
    if os.path.exists(data_root):
        print("Testing data loader...")
        train_loader, val_loader, test_loader = create_bsdcnn_dataloaders(
            data_root=data_root,
            test_patient='PN00',
            batch_size=4,
            num_workers=0,
            window_seconds=10,
            overlap_seconds=5
        )
        
        # Test batch
        for batch_data, batch_labels in train_loader:
            print(f"\nBatch shape: {batch_data.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            print(f"Labels: {batch_labels}")
            break
    else:
        print(f"Data directory not found: {data_root}")
