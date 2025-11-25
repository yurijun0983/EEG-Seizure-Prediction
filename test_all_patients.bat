@echo off
REM Test reverse domain adaptation model on all patients
REM Evaluates cross-patient generalization performance

echo ========================================
echo Cross-Patient Generalization Test
echo Model: Reverse Domain Adaptation
echo ========================================

REM Configuration
set MODEL_PATH=/outputs_reverse_adaptation/best_model.pth
set DATA_ROOT=data/siena-scalp-eeg-database-1.0.0
set PREPROCESSED_DIR=/data/preprocessed_reverse_aligned
set BATCH_SIZE=256
set MODEL_TYPE=standard
set WINDOW_SECONDS=5
set OVERLAP_SECONDS=2.5
set THRESHOLD=0.5

REM Results directory
set RESULTS_DIR=cross_patient_results
mkdir %RESULTS_DIR% 2>nul

echo.
echo Testing model on all patients...
echo Results will be saved to: %RESULTS_DIR%\
echo.

REM Test each patient
for %%p in (PN00 PN03 PN05 PN06 PN07 PN11 PN12 PN13 PN14 PN16 PN17) do (
    echo ----------------------------------------
    echo Testing patient: %%p
    echo ----------------------------------------
    
    REM Run test
    python bsdcnn_test.py ^
      --model_path %MODEL_PATH% ^
      --data_root %DATA_ROOT% ^
      --preprocessed_dir %PREPROCESSED_DIR% ^
      --test_patient %%p ^
      --batch_size %BATCH_SIZE% ^
      --model_type %MODEL_TYPE% ^
      --window_seconds %WINDOW_SECONDS% ^
      --overlap_seconds %OVERLAP_SECONDS% ^
      --threshold %THRESHOLD% ^
      --output_dir %RESULTS_DIR%\%%p
    
    REM Check if test was successful
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Test completed for %%p
    ) else (
        echo ✗ Test failed for %%p
    )
    
    echo.
)

echo ========================================
echo Test completed!
echo Results saved to: %RESULTS_DIR%\
echo ========================================