"""Check if preictal and interictal data are actually different"""
import numpy as np
import os

preprocessed_dir = '/share-data/preprocessed_10s'

print("="*60)
print("Data Quality Check")
print("="*60)

# Load first patient data
patient = 'PN00'
data_file = os.path.join(preprocessed_dir, f'{patient}.npz')

if os.path.exists(data_file):
    data = np.load(data_file)
    preictal = data['preictal']
    interictal = data['interictal']
    
    print(f"\n{patient}:")
    print(f"  Preictal: {len(preictal)} samples, shape: {preictal[0].shape if len(preictal) > 0 else 'N/A'}")
    print(f"  Interictal: {len(interictal)} samples, shape: {interictal[0].shape if len(interictal) > 0 else 'N/A'}")
    
    if len(preictal) > 0 and len(interictal) > 0:
        # Calculate statistics
        pre_mean = np.mean([seg.mean() for seg in preictal[:100]])
        pre_std = np.mean([seg.std() for seg in preictal[:100]])
        pre_var = np.mean([seg.var() for seg in preictal[:100]])
        
        int_mean = np.mean([seg.mean() for seg in interictal[:100]])
        int_std = np.mean([seg.std() for seg in interictal[:100]])
        int_var = np.mean([seg.var() for seg in interictal[:100]])
        
        print(f"\n  Preictal stats (first 100):")
        print(f"    Mean: {pre_mean:.6f}, Std: {pre_std:.6f}, Var: {pre_var:.6f}")
        
        print(f"\n  Interictal stats (first 100):")
        print(f"    Mean: {int_mean:.6f}, Std: {int_std:.6f}, Var: {int_var:.6f}")
        
        print(f"\n  Difference:")
        print(f"    Mean diff: {abs(pre_mean - int_mean):.6f}")
        print(f"    Std diff: {abs(pre_std - int_std):.6f}")
        print(f"    Var diff: {abs(pre_var - int_var):.6f}")
        
        # Check if they're too similar
        if abs(pre_mean - int_mean) < 0.01 and abs(pre_std - int_std) < 0.1:
            print(f"\n  ⚠️  WARNING: Preictal and Interictal are VERY SIMILAR!")
            print(f"  This explains why the model cannot learn to distinguish them.")
        else:
            print(f"\n  ✓ Preictal and Interictal show some differences.")
        
        # Sample a few segments
        print(f"\n  Sample segment comparison:")
        for i in range(min(3, len(preictal), len(interictal))):
            pre_seg = preictal[i]
            int_seg = interictal[i]
            print(f"    Segment {i}: Pre mean={pre_seg.mean():.4f}, Int mean={int_seg.mean():.4f}")
            print(f"              Pre std={pre_seg.std():.4f}, Int std={int_seg.std():.4f}")
            
    data.close()
else:
    print(f"File not found: {data_file}")
