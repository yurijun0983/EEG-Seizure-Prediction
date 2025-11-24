#!/bin/bash
# Test different thresholds

for threshold in 0.35 0.4 0.45
do
  echo "Testing with threshold=$threshold"
  python bsdcnn_test.py \
    --model_path outputs_retrain/best_model.pth \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --preprocessed_dir preprocessed_original \
    --test_patient PN14 \
    --batch_size 256 \
    --model_type standard \
    --window_seconds 5 \
    --overlap_seconds 2.5 \
    --threshold $threshold
  echo "---"
done
