@echo off
REM Run complete reverse domain adaptation pipeline for all patients (leave-one-out cross-validation)

echo ==================================================
echo Complete Reverse Domain Adaptation Pipeline
echo Leave-One-Out Cross-Validation for All Patients
echo ==================================================

REM Configuration
set DATA_ROOT=data/siena-scalp-eeg-database-1.0.0
set PREPROCESSED_DIR=/data/preprocessed_original
set BATCH_SIZE=256
set MODEL_TYPE=standard
set WINDOW_SECONDS=5
set OVERLAP_SECONDS=2.5
set EPOCHS=50
set PATIENCE=10

echo.
echo Processing all patients...
echo.

REM Process each patient as test patient
for %%p in (PN00 PN03 PN05 PN06 PN07 PN11 PN12 PN13 PN14 PN16 PN17) do (
    echo ==================================================
    echo Processing patient: %%p (as test patient)
    echo ==================================================
    
    REM Create patient-specific directories
    set REVERSE_ALIGNED_DIR=/data/preprocessed_reverse_aligned_%%p
    set OUTPUT_DIR=/outputs_reverse_adaptation_%%p
    
    echo Directories:
    echo   Reverse aligned: %REVERSE_ALIGNED_DIR%
    echo   Model output: %OUTPUT_DIR%
    echo.
    
    REM Step 1: Reverse domain adaptation
    echo Step 1: Reverse domain adaptation...
    python reverse_domain_adaptation.py ^
      --preprocessed_dir %PREPROCESSED_DIR% ^
      --target_patient %%p ^
      --output_dir %REVERSE_ALIGNED_DIR% ^
      --data_root %DATA_ROOT%
    
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ Reverse domain adaptation failed for %%p
        goto :next_patient
    )
    echo ✓ Reverse domain adaptation completed
    echo.
    
    REM Step 2: Train model
    echo Step 2: Training model...
    python bsdcnn_train.py ^
      --data_root %DATA_ROOT% ^
      --preprocessed_dir %REVERSE_ALIGNED_DIR% ^
      --test_patient %%p ^
      --output_dir %OUTPUT_DIR% ^
      --epochs %EPOCHS% ^
      --batch_size %BATCH_SIZE% ^
      --model_type %MODEL_TYPE% ^
      --window_seconds %WINDOW_SECONDS% ^
      --overlap_seconds %OVERLAP_SECONDS% ^
      --num_workers 0 ^
      --use_focal_loss ^
      --focal_alpha 0.75 ^
      --patience %PATIENCE%
    
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ Training failed for %%p
        goto :next_patient
    )
    echo ✓ Training completed
    echo.
    
    REM Step 3: Test model
    echo Step 3: Testing model...
    python bsdcnn_test.py ^
      --model_path %OUTPUT_DIR%/best_model.pth ^
      --data_root %DATA_ROOT% ^
      --preprocessed_dir %REVERSE_ALIGNED_DIR% ^
      --test_patient %%p ^
      --batch_size %BATCH_SIZE% ^
      --model_type %MODEL_TYPE% ^
      --window_seconds %WINDOW_SECONDS% ^
      --overlap_seconds %OVERLAP_SECONDS% ^
      --threshold 0.5 ^
      --output_dir %OUTPUT_DIR%
    
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ Testing failed for %%p
        goto :next_patient
    )
    echo ✓ Testing completed
    echo.
    
    :next_patient
    echo.
    echo Completed processing for %%p
    echo.
)

echo ==================================================
echo ALL PATIENTS PROCESSED!
echo ==================================================