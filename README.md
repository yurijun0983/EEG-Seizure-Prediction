# EEG Seizure Prediction with Multi-Objective Optimization

基于深度学习和多目标遗传算法的跨患者癫痫预测系统

## 📌 项目简介

本项目实现了一套完整的EEG癫痫预测系统，结合了深度学习模型（BSDCNN）和NSGA-II多目标优化算法，用于优化源患者选择和通道选择，以提升跨患者癫痫预测性能。

### 核心特性

- 🧠 **多模型架构**: 支持BSDCNN、CNN-BiLSTM、Transformer等多种深度学习模型
- 🔬 **NSGA-II优化**: 多目标遗传算法优化源患者选择，平衡性能、泛化性和数据质量
- 📊 **通道选择**: 遗传算法驱动的EEG通道优化，减少硬件成本
- 🎯 **跨患者预测**: 留一法验证，支持个性化癫痫预测
- 📈 **完整流程**: 数据预处理、模型训练、评估、可视化一体化

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- CUDA >= 11.0 (推荐GPU训练)
- 8GB+ RAM (NSGA-II优化需要)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载Siena Scalp EEG数据集
2. 解压到 `data/siena-scalp-eeg-database-1.0.0/`
3. 运行预处理:

```bash
python bsdcnn_preprocess.py \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --output_dir preprocessed_data \
    --window_seconds 5 \
    --overlap_seconds 2.5
```

### 基础训练

```bash
python bsdcnn_train.py \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --preprocessed_dir preprocessed_data \
    --test_patient PN14 \
    --batch_size 256 \
    --epochs 100 \
    --use_weighted_loss \
    --num_workers 0
```

## 🎯 核心功能

### 1. NSGA-II多目标患者选择优化

同时优化三个目标：
- 最大化F1 Score (预测性能)
- 最小化患者数量 (提高泛化性)
- 最小化样本不平衡度 (数据质量)

**运行NSGA-II优化**:

```bash
python bsdcnn_train.py \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --preprocessed_dir preprocessed_data \
    --test_patient PN14 \
    --ga_ps_optimize \
    --ga_ps_population 30 \
    --ga_ps_generations 20 \
    --ga_train_epochs 8 \
    --ga_ps_only \
    --batch_size 256 \
    --num_workers 0 \
    --output_dir outputs_nsga2_pn14
```

**可视化结果**:

```bash
python visualize_nsga2_results.py \
    --results_path outputs_nsga2_pn14/nsga2_patient_selection_results.json \
    --output_dir outputs_nsga2_pn14/visualizations
```

### 2. 遗传算法通道选择

从29个标准EEG通道中选择最优子集：

```bash
python run_channel_selection_tutorial.py --use-model
```

### 3. 模型评估与测试

```bash
python bsdcnn_test.py \
    --model_path outputs/best_model.pth \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --test_patient PN14 \
    --model_type standard
```

### 4. 阈值优化

```bash
python find_optimal_threshold.py \
    --model_path outputs/best_model.pth \
    --data_root data/siena-scalp-eeg-database-1.0.0 \
    --model_type standard \
    --metric f1
```

## 📊 项目结构

```
ST-WGAN-GP-Bi-LSTM/
├── data/                           # 数据目录
│   └── siena-scalp-eeg-database-1.0.0/
├── bsdcnn_data_loader.py          # 数据加载器
├── bsdcnn_model.py                # 模型定义
├── bsdcnn_train.py                # 训练脚本
├── bsdcnn_test.py                 # 测试脚本
├── bsdcnn_preprocess.py           # 数据预处理
├── ga_patient_selection.py        # NSGA-II患者选择
├── ga_channel_selection_with_model.py  # 通道选择
├── visualize_nsga2_results.py     # 结果可视化
├── focal_loss.py                  # Focal Loss实现
├── eeg_augmentation.py            # 数据增强
└── requirements.txt               # 依赖列表
```

## 🔬 算法原理

### NSGA-II多目标优化

NSGA-II (Non-dominated Sorting Genetic Algorithm II) 是一种高效的多目标进化算法，通过以下机制优化患者选择：

1. **非支配排序**: 将种群按帕累托层级分层
2. **拥挤距离**: 保持解的多样性
3. **精英保留**: 优秀个体跨代传递
4. **帕累托前沿**: 生成多个最优解供选择

### 模型架构

- **BSDCNN**: Binary Single-Dimensional CNN，轻量级二值化卷积网络
- **Attention-CNN-BiLSTM**: 结合注意力机制的混合模型
- **Transformer**: 基于自注意力的时序建模

## 🛠️ 技术栈

- **深度学习**: PyTorch >= 2.0.0
- **数据处理**: NumPy, Pandas, SciPy
- **EEG处理**: pyedflib, MNE
- **可视化**: Matplotlib, Seaborn
- **优化**: scikit-learn, DEAP

## ⚙️ 配置说明

### 关键参数

**训练参数**:
- `--batch_size`: 批次大小 (推荐256)
- `--epochs`: 训练轮数 (默认100)
- `--lr`: 学习率 (默认0.001)
- `--use_weighted_loss`: 启用加权损失处理类别不平衡

**NSGA-II参数**:
- `--ga_ps_population`: 种群大小 (推荐20-30)
- `--ga_ps_generations`: 进化代数 (推荐15-20)
- `--ga_train_epochs`: 每代训练轮数 (推荐5-8)
- `--ga_ps_mode`: 选择模式 (binary/weight)

**系统参数**:
- `--num_workers`: 数据加载进程数 (Windows必须为0)

## 🐛 常见问题

### Q1: CUDA out of memory

降低batch_size或使用CPU训练：
```bash
--batch_size 64  # 或更小
```

### Q2: FileNotFoundError

确保数据已预处理：
```bash
python bsdcnn_preprocess.py --data_root data/siena-scalp-eeg-database-1.0.0 --output_dir preprocessed_data
```



## 📝 待办事项

- [ ] 添加更多预训练模型
- [ ] 支持实时预测API
- [ ] 集成更多数据集
- [ ] 优化内存使用
- [ ] 添加模型压缩

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅供学术研究使用。

## 🙏 致谢

感谢Siena Scalp EEG数据集提供者以及所有相关研究论文的作者。

---

⭐ 如果这个项目对您有帮助，请给个Star！
