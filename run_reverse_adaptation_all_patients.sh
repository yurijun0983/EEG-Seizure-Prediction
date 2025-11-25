#!/bin/bash
# Run complete reverse domain adaptation pipeline for all patients (leave-one-out cross-validation)

echo "=================================================="
echo "Complete Reverse Domain Adaptation Pipeline"
echo "Leave-One-Out Cross-Validation for All Patients"
echo "=================================================="

# Configuration
DATA_ROOT="data/siena-scalp-eeg-database-1.0.0"
PREPROCESSED_DIR="/data/preprocessed_original"
BATCH_SIZE=256
MODEL_TYPE="standard"
WINDOW_SECONDS=5
OVERLAP_SECONDS=2.5
EPOCHS=50
PATIENCE=10

# All patients
PATIENTS=("PN00" "PN03" "PN05" "PN06" "PN07" "PN11" "PN12" "PN13" "PN14" "PN16" "PN17")

# Skip preprocessing if already done
SKIP_PREPROCESS=true

echo ""
echo "Patients to process: ${PATIENTS[@]}"
echo ""

# Process each patient as test patient
for test_patient in "${PATIENTS[@]}"
do
    echo "=================================================="
    echo "Processing patient: $test_patient (as test patient)"
    echo "=================================================="
    
    # Create patient-specific directories
    REVERSE_ALIGNED_DIR="/data/preprocessed_reverse_aligned_${test_patient}"
    OUTPUT_DIR="/outputs_reverse_adaptation_${test_patient}"
    
    echo "Directories:"
    echo "  Reverse aligned: $REVERSE_ALIGNED_DIR"
    echo "  Model output: $OUTPUT_DIR"
    echo ""
    
    # Step 1: Reverse domain adaptation
    echo "Step 1: Reverse domain adaptation..."
    if [ "$SKIP_PREPROCESS" = true ]; then
        python reverse_domain_adaptation.py \
          --preprocessed_dir $PREPROCESSED_DIR \
          --target_patient $test_patient \
          --output_dir $REVERSE_ALIGNED_DIR \
          --data_root $DATA_ROOT
    else
        python reverse_domain_adaptation.py \
          --preprocessed_dir $PREPROCESSED_DIR \
          --target_patient $test_patient \
          --output_dir $REVERSE_ALIGNED_DIR \
          --data_root $DATA_ROOT
    fi
    
    if [ $? -ne 0 ]; then
        echo "✗ Reverse domain adaptation failed for $test_patient"
        continue
    fi
    echo "✓ Reverse domain adaptation completed"
    echo ""
    
    # Step 2: Train model
    echo "Step 2: Training model..."
    python bsdcnn_train.py \
      --data_root $DATA_ROOT \
      --preprocessed_dir $REVERSE_ALIGNED_DIR \
      --test_patient $test_patient \
      --output_dir $OUTPUT_DIR \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --model_type $MODEL_TYPE \
      --window_seconds $WINDOW_SECONDS \
      --overlap_seconds $OVERLAP_SECONDS \
      --num_workers 0 \
      --use_focal_loss \
      --focal_alpha 0.75 \
      --patience $PATIENCE
    
    if [ $? -ne 0 ]; then
        echo "✗ Training failed for $test_patient"
        continue
    fi
    echo "✓ Training completed"
    echo ""
    
    # Step 3: Test model
    echo "Step 3: Testing model..."
    python bsdcnn_test.py \
      --model_path ${OUTPUT_DIR}/best_model.pth \
      --data_root $DATA_ROOT \
      --preprocessed_dir $REVERSE_ALIGNED_DIR \
      --test_patient $test_patient \
      --batch_size $BATCH_SIZE \
      --model_type $MODEL_TYPE \
      --window_seconds $WINDOW_SECONDS \
      --overlap_seconds $OVERLAP_SECONDS \
      --threshold 0.5 \
      --output_dir $OUTPUT_DIR
    
    if [ $? -ne 0 ]; then
        echo "✗ Testing failed for $test_patient"
        continue
    fi
    echo "✓ Testing completed"
    echo ""
    
    # Extract and display results
    if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
        F1=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['f1'])")
        RECALL=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['recall'])")
        AUC=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['auc'])")
        echo "Results for $test_patient:"
        echo "  F1: $F1"
        echo "  Recall: $RECALL"
        echo "  AUC: $AUC"
    fi
    
    echo ""
    echo "Completed processing for $test_patient"
    echo ""
done

echo "=================================================="
echo "ALL PATIENTS PROCESSED!"
echo "=================================================="
echo ""
echo "Summary of results:"
echo "Patient,Status,F1,Recall,AUC"
for test_patient in "${PATIENTS[@]}"
do
    OUTPUT_DIR="/outputs_reverse_adaptation_${test_patient}"
    if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
        F1=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['f1'])")
        RECALL=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['recall'])")
        AUC=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/test_results.json'))['metrics']['auc'])")
        echo "$test_patient,Completed,$F1,$RECALL,$AUC"
    else
        echo "$test_patient,Failed,0.0000,0.0000,0.0000"
    fi
done