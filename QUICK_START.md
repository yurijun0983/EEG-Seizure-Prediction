# 快速开始指南

## 1. 一键运行完整流程

```bash
# Windows (PowerShell)
python run_complete_pipeline.py --data_root data/siena-scalp-eeg-database-1.0.0 --test_patient PN14

# Linux/Mac
python run_complete_pipeline.py \
  --data_root data/siena-scalp-eeg-database-1.0.0 \
  --test_patient PN14
```

## 2. 使用脚本文件

### Windows
```cmd
run_pipeline_example.bat
```

### Linux/Mac
```bash
chmod +x run_pipeline_example.sh
./run_pipeline_example.sh
```

## 3. 主要输出

运行完成后,查看以下文件:

- **最终测试结果**: `outputs_final_model/test_results.json`
- **锚点信息**: `outputs_individual_models/best_source_model.json`
- **训练历史**: `outputs_final_model/results.json`

## 4. 查看结果示例

```python
import json

# 查看最终测试结果
with open('outputs_final_model/test_results.json', 'r') as f:
    results = json.load(f)
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"AUC: {results['metrics']['auc']:.4f}")

# 查看锚点患者
with open('outputs_individual_models/best_source_model.json', 'r') as f:
    anchor = json.load(f)
    print(f"Anchor Patient: {anchor['anchor_patient']}")
    print(f"F1 on Test Patient: {anchor['anchor_metrics']['f1']:.4f}")
```

## 5. 常见参数调整

### 加速训练
```bash
python run_complete_pipeline.py \
  --data_root data/siena-scalp-eeg-database-1.0.0 \
  --test_patient PN14 \
  --epochs_individual 30 \
  --epochs_final 50 \
  --batch_size 512
```

### 使用已有数据(跳过预处理)
```bash
python run_complete_pipeline.py \
  --data_root data/siena-scalp-eeg-database-1.0.0 \
  --test_patient PN14 \
  --skip_preprocess
```

### 手动指定锚点
```bash
python run_complete_pipeline.py \
  --data_root data/siena-scalp-eeg-database-1.0.0 \
  --test_patient PN14 \
  --skip_find_anchor \
  --anchor_patient PN11
```

## 6. 留一交叉验证(所有患者)

### Windows
```powershell
foreach ($patient in @('PN00','PN03','PN05','PN06','PN07','PN11','PN12','PN13','PN14','PN16','PN17')) {
  python run_complete_pipeline.py `
    --data_root data/siena-scalp-eeg-database-1.0.0 `
    --test_patient $patient `
    --skip_preprocess `
    --final_output_dir outputs_loo_$patient
}
```

### Linux/Mac
```bash
for patient in PN00 PN03 PN05 PN06 PN07 PN11 PN12 PN13 PN14 PN16 PN17
do
  python run_complete_pipeline.py \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --test_patient $patient \
    --skip_preprocess \
    --final_output_dir outputs_loo_${patient}
done
```

## 7. 故障排除

### 内存不足
```bash
# 减小batch size
--batch_size 64
```

### CUDA错误
```bash
# 使用CPU
export CUDA_VISIBLE_DEVICES=""
python run_complete_pipeline.py ...
```

### 进度查看
所有步骤都会打印详细进度,检查终端输出即可

## 8. 预期运行时间

| 步骤 | GPU | CPU |
|------|-----|-----|
| 预处理 | 10-20分钟 | 10-20分钟 |
| 个体化训练(11个患者) | 2-4小时 | 8-12小时 |
| 找锚点 | 10-20分钟 | 30-60分钟 |
| 域对齐 | 1-2分钟 | 1-2分钟 |
| 最终训练 | 30-60分钟 | 2-4小时 |
| **总计** | **3-5小时** | **11-17小时** |

## 9. 更多详情

查看完整文档: `DOMAIN_ADAPTATION_README.md`
