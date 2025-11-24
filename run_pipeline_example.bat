@echo off
REM Example script for running the complete domain adaptation pipeline (Windows)
REM Test patient: PN14

echo ========================================
echo Domain Adaptation Pipeline for PN14
echo ========================================

REM Configuration
set DATA_ROOT=data/siena-scalp-eeg-database-1.0.0
set TEST_PATIENT=PN14
set PREPROCESSED_ORIGINAL=preprocessed_original
set PREPROCESSED_ALIGNED=preprocessed_aligned
set INDIVIDUAL_MODELS=outputs_individual_models
set FINAL_OUTPUT=outputs_final_model

set EPOCHS_INDIVIDUAL=50
set EPOCHS_FINAL=100
set BATCH_SIZE=256
set MODEL_TYPE=standard

echo.
echo Step 0: Preprocessing raw data...
python bsdcnn_preprocess.py ^
  --data_root %DATA_ROOT% ^
  --output_dir %PREPROCESSED_ORIGINAL% ^
  --window_seconds 5 ^
  --overlap_seconds 2.5

echo.
echo Step 1: Training individual patient models...
python train_individual_models.py ^
  --data_root %DATA_ROOT% ^
  --preprocessed_dir %PREPROCESSED_ORIGINAL% ^
  --output_dir %INDIVIDUAL_MODELS% ^
  --epochs %EPOCHS_INDIVIDUAL% ^
  --batch_size %BATCH_SIZE% ^
  --model_type %MODEL_TYPE%

echo.
echo Step 2: Finding best source model (anchor)...
python find_best_source_model.py ^
  --data_root %DATA_ROOT% ^
  --preprocessed_dir %PREPROCESSED_ORIGINAL% ^
  --models_dir %INDIVIDUAL_MODELS% ^
  --test_patient %TEST_PATIENT% ^
  --batch_size %BATCH_SIZE% ^
  --model_type %MODEL_TYPE%

REM Extract anchor patient from results (manual for now)
echo.
echo Please check %INDIVIDUAL_MODELS%\best_source_model.json for anchor patient
echo Press any key to continue after noting the anchor patient...
pause

set /p ANCHOR_PATIENT=Enter anchor patient ID (e.g., PN11): 

echo.
echo Step 3: Domain adaptation (Euclidean Alignment)...
python domain_adaptation.py ^
  --preprocessed_dir %PREPROCESSED_ORIGINAL% ^
  --anchor_patient %ANCHOR_PATIENT% ^
  --test_patient %TEST_PATIENT% ^
  --output_dir %PREPROCESSED_ALIGNED% ^
  --data_root %DATA_ROOT%

echo.
echo Step 4: Training final model with aligned data...
python bsdcnn_train.py ^
  --data_root %DATA_ROOT% ^
  --preprocessed_dir %PREPROCESSED_ALIGNED% ^
  --test_patient %TEST_PATIENT% ^
  --output_dir %FINAL_OUTPUT% ^
  --epochs %EPOCHS_FINAL% ^
  --batch_size %BATCH_SIZE% ^
  --lr 0.001 ^
  --model_type %MODEL_TYPE% ^
  --window_seconds 5 ^
  --overlap_seconds 2.5 ^
  --num_workers 0 ^
  --use_focal_loss

echo.
echo Step 5: Final evaluation on test patient...
python bsdcnn_test.py ^
  --model_path %FINAL_OUTPUT%\best_model.pth ^
  --data_root %DATA_ROOT% ^
  --preprocessed_dir %PREPROCESSED_ORIGINAL% ^
  --test_patient %TEST_PATIENT% ^
  --batch_size %BATCH_SIZE% ^
  --model_type %MODEL_TYPE% ^
  --window_seconds 5 ^
  --overlap_seconds 2.5 ^
  --output_dir %FINAL_OUTPUT%

echo.
echo ========================================
echo Pipeline completed!
echo ========================================
echo Results:
echo   - Individual models: %INDIVIDUAL_MODELS%
echo   - Anchor patient: %ANCHOR_PATIENT%
echo   - Aligned data: %PREPROCESSED_ALIGNED%
echo   - Final model: %FINAL_OUTPUT%
echo   - Test results: %FINAL_OUTPUT%\test_results.json
pause
