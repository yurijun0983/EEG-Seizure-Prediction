#!/bin/bash
# Test reverse domain adaptation model on all patients
# Evaluates cross-patient generalization performance

echo "========================================"
echo "Cross-Patient Generalization Test"
echo "Model: Reverse Domain Adaptation"
echo "========================================"

# Configuration
MODEL_PATH="/outputs_reverse_adaptation/best_model.pth"
DATA_ROOT="data/siena-scalp-eeg-database-1.0.0"
PREPROCESSED_DIR="/data/preprocessed_reverse_aligned"
BATCH_SIZE=256
MODEL_TYPE="standard"
WINDOW_SECONDS=5
OVERLAP_SECONDS=2.5
THRESHOLD=0.5

# All patients except PN09 (if it doesn't exist in your dataset)
PATIENTS=("PN00" "PN03" "PN05" "PN06" "PN07" "PN11" "PN12" "PN13" "PN14" "PN16" "PN17")

# Results directory
RESULTS_DIR="cross_patient_results"
mkdir -p $RESULTS_DIR

echo ""
echo "Testing model on all patients..."
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Test each patient
for patient in "${PATIENTS[@]}"
do
    echo "----------------------------------------"
    echo "Testing patient: $patient"
    echo "----------------------------------------"
    
    # Run test
    python bsdcnn_test.py \
      --model_path $MODEL_PATH \
      --data_root $DATA_ROOT \
      --preprocessed_dir $PREPROCESSED_DIR \
      --test_patient $patient \
      --batch_size $BATCH_SIZE \
      --model_type $MODEL_TYPE \
      --window_seconds $WINDOW_SECONDS \
      --overlap_seconds $OVERLAP_SECONDS \
      --threshold $THRESHOLD \
      --output_dir "${RESULTS_DIR}/${patient}"
    
    # Check if test was successful
    if [ $? -eq 0 ]; then
        echo "✓ Test completed for $patient"
        
        # Extract key metrics
        if [ -f "${RESULTS_DIR}/${patient}/test_results.json" ]; then
            F1=$(python -c "import json; print(json.load(open('${RESULTS_DIR}/${patient}/test_results.json'))['metrics']['f1'])")
            RECALL=$(python -c "import json; print(json.load(open('${RESULTS_DIR}/${patient}/test_results.json'))['metrics']['recall'])")
            AUC=$(python -c "import json; print(json.load(open('${RESULTS_DIR}/${patient}/test_results.json'))['metrics']['auc'])")
            echo "  F1: $F1, Recall: $RECALL, AUC: $AUC"
        fi
    else
        echo "✗ Test failed for $patient"
    fi
    
    echo ""
done

# Generate summary report
echo "========================================"
echo "SUMMARY REPORT"
echo "========================================"

echo "Patient,F1,Recall,Precision,Specificity,AUC" > "${RESULTS_DIR}/summary.csv"

for patient in "${PATIENTS[@]}"
do
    if [ -f "${RESULTS_DIR}/${patient}/test_results.json" ]; then
        # Extract metrics using Python
        METRICS=$(python -c "
import json
try:
    data = json.load(open('${RESULTS_DIR}/${patient}/test_results.json'))
    metrics = data['metrics']
    print(f'{metrics[\"f1\"]:.4f},{metrics[\"recall\"]:.4f},{metrics[\"precision\"]:.4f},{metrics[\"specificity\"]:.4f},{metrics[\"auc\"]:.4f}')
except:
    print('0.0000,0.0000,0.0000,0.0000,0.0000')
")
        echo "$patient,$METRICS" >> "${RESULTS_DIR}/summary.csv"
        echo "$patient: $METRICS"
    else
        echo "$patient: No results"
        echo "$patient,0.0000,0.0000,0.0000,0.0000,0.0000" >> "${RESULTS_DIR}/summary.csv"
    fi
done

echo ""
echo "Summary saved to: ${RESULTS_DIR}/summary.csv"
echo ""
echo "Test completed!"