"""
Training script for BSDCNN (Binary Single-Dimensional CNN) for Seizure Prediction
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from bsdcnn_model import BSDCNN, BSDCNNSimplified, create_bsdcnn_model
from bsdcnn_data_loader import create_bsdcnn_dataloaders
from focal_loss import FocalLoss, CombinedLoss
from eeg_augmentation import EEGAugmentation, ga_generate_augmented_batch
from ga_patient_selection import run_nsga2_patient_selection


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': 0.0,
        'auc': 0.0
    }
    
    # Calculate specificity
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    
    # Calculate AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_data, batch_labels in pbar:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(val_loader, desc='Validating'):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Statistics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics


def train_model(args):
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup tensorboard
    log_dir = os.path.join(args.output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with augmentation for training
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    train_loader, val_loader, test_loader = create_bsdcnn_dataloaders(
        data_root=args.data_root,
        test_patient=args.test_patient,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        val_split=args.val_split,
        preprocessed_dir=args.preprocessed_dir,  # Add this parameter
        use_augmentation=True  # Enable data augmentation for training
    )
    
    # Get input dimensions from first batch
    sample_batch, _ = next(iter(train_loader))
    num_channels = sample_batch.shape[1]
    sequence_length = sample_batch.shape[2]
    
    print(f"\nInput shape: (batch_size, {num_channels}, {sequence_length})")
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    # Prepare model kwargs based on model type
    model_kwargs = {
        'num_channels': num_channels,
        'sequence_length': sequence_length,
        'num_classes': 2,
    }
    
    # Only add use_binary_activation for binary models
    if args.model_type in ['full', 'simplified']:
        model_kwargs['use_binary_activation'] = args.use_binary_activation
    
    model = create_bsdcnn_model(
        model_type=args.model_type,
        **model_kwargs
    )
    
    model = model.to(device)
    
    print(f"Model type: {args.model_type}")
    if hasattr(model, 'count_parameters'):
        print(f"Total parameters: {model.count_parameters():,}")
        print(f"Model size: {model.get_model_size():.2f} MB")
    
    # Loss function and optimizer
    if args.use_focal_loss:
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.use_weighted_loss:
        # Calculate class weights
        n_train = len(train_loader.dataset)
        n_positive = sum([label.item() for _, label in train_loader.dataset])
        n_negative = n_train - n_positive
        
        weight = torch.FloatTensor([1.0, n_negative / n_positive]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        print(f"Using weighted loss with weights: {weight.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # ========== 可选：GA训练循环（遗传算法直接优化训练数据交叉组合） ==========
    if getattr(args, 'ga_augment', False):
        print("\n" + "="*50)
        print("启用遗传算法交叉训练（GA-based Crossover Training）")
        print("直接从npz数据集进行发作间期与发作期的交叉，迭代优化")
        print("="*50)
        
        # 从npz文件直接加载发作间期和发作期数据
        print("\n从预处理的npz文件加载发作间期和发作期原始数据...")
        if not args.preprocessed_dir:
            raise ValueError("启用GA训练需要指定 --preprocessed_dir 参数")
        
        # 获取训练患者列表
        if args.test_patient:
            all_patients = sorted([d for d in os.listdir(args.data_root) 
                                 if os.path.isdir(os.path.join(args.data_root, d)) and d.startswith('PN')])
            train_patients = [p for p in all_patients if p != args.test_patient]
        else:
            raise ValueError("GA训练模式需要指定 --test_patient")
        
        # 加载所有训练患者的原始npz数据
        all_preictal = []
        all_interictal = []
        for patient_id in train_patients:
            npz_path = os.path.join(args.preprocessed_dir, f'{patient_id}.npz')
            if os.path.exists(npz_path):
                print(f"  加载 {patient_id}...")
                data = np.load(npz_path)
                all_preictal.extend(list(data['preictal']))
                all_interictal.extend(list(data['interictal']))
                data.close()
        
        print(f"\n加载完成: {len(all_preictal)} 个发作期片段, {len(all_interictal)} 个发作间期片段")
        
        # 初始化GA种群：每个个体代表一种交叉策略
        # 参考文献:
        # - GA用于时序数据增强 (Frontiers in Neuroscience, 2023)
        # - 交叉操作优化训练集 (IEEE Sensors, 2022)
        population_size = 6
        population = [
            {'type': 'window', 'lambda': 0.3, 'crossover_ratio': 0.3},
            {'type': 'channel', 'lambda': 0.5, 'crossover_ratio': 0.5},
            {'type': 'time',   'lambda': 0.7, 'crossover_ratio': 0.4, 'time_mask_density': 0.2},
            {'type': 'window', 'lambda': 0.6, 'crossover_ratio': 0.6},
            {'type': 'channel', 'lambda': 0.4, 'crossover_ratio': 0.3},
            {'type': 'time',   'lambda': 0.5, 'crossover_ratio': 0.5, 'time_mask_density': 0.3},
        ]
        
        elite_size = 2
        mutation_sigma = 0.1
        ga_generations = getattr(args, 'ga_generations', 5)
        train_epochs_per_gen = getattr(args, 'ga_train_epochs', 10)  # 每代训练的epoch数
        
        best_individual = None
        best_f1 = -1.0
        fitness_history = []
        
        def create_crossover_dataset(individual, preictal_data, interictal_data):
            """
            根据个体参数创建交叉后的训练数据集
            individual: GA个体（交叉策略参数）
            返回: 交叉后的训练segments和labels
            """
            crossover_ratio = individual.get('crossover_ratio', 0.5)
            n_crossover = int(len(preictal_data) * crossover_ratio)
            
            # 生成交叉样本
            crossover_segs, crossover_labels = ga_generate_augmented_batch(
                preictal_data, interictal_data, individual, batch_size=n_crossover
            )
            
            # 合并：原始发作期 + 原始发作间期 + 交叉样本
            all_segments = list(preictal_data) + list(interictal_data) + crossover_segs
            all_labels = [1] * len(preictal_data) + [0] * len(interictal_data) + crossover_labels
            
            return all_segments, all_labels
        
        def train_and_evaluate_individual(individual):
            """
            用个体的交叉策略生成训练集，训练模型，返回验证F1
            """
            # 生成训练数据
            train_segs, train_labels = create_crossover_dataset(
                individual, all_preictal, all_interictal
            )
            
            # 创建数据集和dataloader
            from bsdcnn_data_loader import BSDCNNDataset
            from sklearn.model_selection import train_test_split
            
            # 划分训练/验证
            train_segs_split, val_segs_split, train_labels_split, val_labels_split = train_test_split(
                train_segs, train_labels, test_size=args.val_split, 
                stratify=train_labels, random_state=42
            )
            
            train_ds = BSDCNNDataset(train_segs_split, train_labels_split, augment=False)
            val_ds = BSDCNNDataset(val_segs_split, val_labels_split, augment=False)
            
            from torch.utils.data import DataLoader
            train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            
            # 创建新模型
            temp_model = create_bsdcnn_model(
                model_type=args.model_type,
                num_channels=num_channels,
                sequence_length=sequence_length,
                num_classes=2,
            ).to(device)
            temp_optimizer = optim.Adam(temp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # 训练指定epoch数
            for epoch in range(train_epochs_per_gen):
                _ = train_epoch(temp_model, train_ld, criterion, temp_optimizer, device)
            
            # 验证
            val_metrics = validate(temp_model, val_ld, criterion, device)
            
            # 清理
            del temp_model, temp_optimizer, train_ds, val_ds, train_ld, val_ld
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            return val_metrics['f1']
        
        # 开始GA迭代
        print(f"\n开始遗传算法迭代训练（{ga_generations} 代，每代训练 {train_epochs_per_gen} epochs）...")
        print("="*70)
        
        for gen in range(ga_generations):
            print(f"\n【第 {gen+1}/{ga_generations} 代】")
            print("-" * 70)
            
            # 评估种群中每个个体
            fitness = []
            for i, individual in enumerate(population):
                print(f"\n  个体 {i+1}/{len(population)}: type={individual['type']}, "
                      f"lambda={individual['lambda']:.2f}, "
                      f"crossover_ratio={individual.get('crossover_ratio', 0.5):.2f}")
                
                f1 = train_and_evaluate_individual(individual)
                fitness.append(f1)
                print(f"    → 验证F1 = {f1:.4f}")
            
            # 记录本代最优
            gen_best_idx = np.argmax(fitness)
            gen_best_f1 = fitness[gen_best_idx]
            gen_best_individual = population[gen_best_idx]
            
            fitness_history.append({
                'generation': gen + 1,
                'best_f1': gen_best_f1,
                'avg_f1': np.mean(fitness),
                'best_individual': gen_best_individual.copy()
            })
            
            print(f"\n  本代最优: F1={gen_best_f1:.4f}, 平均F1={np.mean(fitness):.4f}")
            
            # 更新全局最优
            if gen_best_f1 > best_f1:
                best_f1 = gen_best_f1
                best_individual = gen_best_individual.copy()
                print(f"  ★★★ 刷新全局最优! F1={best_f1:.4f} ★★★")
                print(f"      最优策略: {best_individual}")
            
            # 如果不是最后一代，进行选择、交叉、变异
            if gen < ga_generations - 1:
                # 精英保留
                elite_idx = np.argsort(fitness)[-elite_size:]
                elite = [population[i].copy() for i in elite_idx]
                
                # 锦标赛选择
                selected = []
                for _ in range(len(population)):
                    idx = np.random.choice(len(population), size=min(3, len(population)), replace=False)
                    winner_idx = idx[np.argmax([fitness[i] for i in idx])]
                    selected.append(population[winner_idx].copy())
                
                # 交叉与变异生成新种群
                new_population = elite.copy()
                while len(new_population) < len(population):
                    # 选择父代
                    pa, pb = np.random.choice(len(selected), size=2, replace=False)
                    parent_a = selected[pa]
                    parent_b = selected[pb]
                    
                    # 交叉
                    child = parent_a.copy()
                    if np.random.rand() < 0.8:
                        child['lambda'] = parent_b['lambda']
                        child['type'] = np.random.choice([parent_a['type'], parent_b['type']])
                        child['crossover_ratio'] = (parent_a.get('crossover_ratio', 0.5) + 
                                                   parent_b.get('crossover_ratio', 0.5)) / 2.0
                    
                    # 变异
                    if np.random.rand() < 0.3:
                        child['lambda'] = float(np.clip(child['lambda'] + np.random.randn() * mutation_sigma, 0.05, 0.95))
                    if np.random.rand() < 0.3:
                        child['crossover_ratio'] = float(np.clip(child.get('crossover_ratio', 0.5) + 
                                                                np.random.randn() * 0.1, 0.1, 0.9))
                    if np.random.rand() < 0.2 and child['type'] == 'time':
                        child['time_mask_density'] = float(np.clip(child.get('time_mask_density', 0.2) + 
                                                                  np.random.randn() * 0.1, 0.0, 0.8))
                    
                    new_population.append(child)
                
                population = new_population[:len(population)]
                print(f"\n  新种群已生成，进入下一代...")
        
        # GA迭代完成
        print("\n" + "="*70)
        print("【GA训练完成】")
        print("="*70)
        print(f"全局最优F1: {best_f1:.4f}")
        print(f"最优交叉策略: {best_individual}")
        
        # 保存GA历史
        ga_history_path = os.path.join(args.output_dir, 'ga_history.json')
        with open(ga_history_path, 'w') as f:
            json.dump({
                'best_f1': best_f1,
                'best_individual': best_individual,
                'fitness_history': fitness_history
            }, f, indent=4)
        print(f"\nGA历史已保存到: {ga_history_path}")
        
        # 用最优策略重新生成完整训练集并训练最终模型
        print("\n使用最优策略生成最终训练集并训练完整模型...")
        final_train_segs, final_train_labels = create_crossover_dataset(
            best_individual, all_preictal, all_interictal
        )
        
        from bsdcnn_data_loader import BSDCNNDataset
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        
        train_segs_final, val_segs_final, train_labels_final, val_labels_final = train_test_split(
            final_train_segs, final_train_labels, test_size=args.val_split,
            stratify=final_train_labels, random_state=42
        )
        
        train_dataset = BSDCNNDataset(train_segs_final, train_labels_final, augment=False)
        val_dataset = BSDCNNDataset(val_segs_final, val_labels_final, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        print(f"最终训练集: {len(train_dataset)} 样本 (包含{int(len(all_preictal)*best_individual.get('crossover_ratio', 0.5))}个交叉样本)")
        print(f"验证集: {len(val_dataset)} 样本")
    # ========== GA训练循环结束 ==========
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Tensorboard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_f1 = val_metrics['f1']
            
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': best_val_f1,
                'args': vars(args)
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'args': vars(args),
        'training_history': training_history,
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {results_path}")
    
    writer.close()
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train BSDCNN for seizure prediction')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                       help='Directory with preprocessed .npz files (optional)')
    parser.add_argument('--test_patient', type=str, default=None,
                       help='Patient ID for testing (leave-one-out)')
    parser.add_argument('--window_seconds', type=float, default=5,
                       help='Window size in seconds (default: 5s per papers)')
    parser.add_argument('--overlap_seconds', type=float, default=2.5,
                       help='Overlap in seconds (default: 2.5s = 50%% overlap)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='full',
                       choices=['full', 'simplified', 'standard', 'attention_cnn_bilstm', 'transformer'],
                       help='Model architecture type')
    parser.add_argument('--use_binary_activation', action='store_true',
                       help='Use binary activation functions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--use_weighted_loss', action='store_true',
                       help='Use weighted loss for class imbalance')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use Focal Loss instead of weighted CE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal Loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma parameter')
    parser.add_argument('--strong_augmentation', action='store_true',
                       help='Use stronger augmentation for better cross-patient generalization')
    
    # GA training parameters (genetic algorithm for crossover-based training)
    parser.add_argument('--ga_augment', action='store_true',
                       help='启用遗传算法交叉训练(从npz数据集直接交叉发作间期与发作期,迭代优化)')
    parser.add_argument('--ga_generations', type=int, default=5,
                       help='GA迭代代数(每代用不同交叉策略训练模型并评估)')
    parser.add_argument('--ga_train_epochs', type=int, default=10,
                       help='GA每代训练的epoch数(用于快速评估个体适应度)')
    
    # NSGA-II Patient Selection parameters (新增)
    parser.add_argument('--ga_ps_optimize', action='store_true',
                       help='启用NSGA-II多目标患者选择优化')
    parser.add_argument('--ga_ps_population', type=int, default=20,
                       help='NSGA-II种群大小')
    parser.add_argument('--ga_ps_generations', type=int, default=15,
                       help='NSGA-II进化代数')
    parser.add_argument('--ga_ps_mode', type=str, default='binary',
                       choices=['binary', 'weight'],
                       help='患者选择模式: binary=二进制选择, weight=权重优化')
    parser.add_argument('--ga_ps_use_test', action='store_true',
                       help='在测试集(PN14)上评估适应度(谨慎使用,会牺牲严格性)')
    parser.add_argument('--ga_ps_only', action='store_true',
                       help='仅运行NSGA-II优化,不进行后续全量训练')
    
    # System parameters
    parser.add_argument('--output_dir', type=str, default='outputs_bsdcnn',
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("Configuration")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    # ========== NSGA-II Patient Selection Optimization ==========
    if args.ga_ps_optimize:
        # 定义源患者列表 (除了test_patient)
        all_patients = ['PN00', 'PN03', 'PN05', 'PN06', 'PN07', 
                       'PN11', 'PN12', 'PN13', 'PN16', 'PN17']
        
        if args.test_patient:
            source_patients = [p for p in all_patients if p != args.test_patient]
        else:
            raise ValueError("NSGA-II优化需要指定 --test_patient")
        
        print("\n" + "="*70)
        print("🧬 NSGA-II Multi-Objective Patient Selection Optimization")
        print("="*70)
        
        # 设备配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 运行NSGA-II优化
        nsga2_results = run_nsga2_patient_selection(
            source_patients=source_patients,
            args=args,
            device=device,
            output_dir=args.output_dir
        )
        
        # 如果只运行优化,不进行后续训练
        if args.ga_ps_only:
            print("\n" + "="*70)
            print("NSGA-II优化完成! (--ga_ps_only模式,跳过后续训练)")
            print("="*70)
            return
        
        # 使用推荐的患者组合进行全量训练
        print("\n" + "="*70)
        print("使用NSGA-II推荐的患者组合进行全量训练...")
        print("="*70)
        
        recommended_patients = nsga2_results['recommended_solution']['selected_patients']
        print(f"\n推荐的源患者组合: {recommended_patients}")
        print(f"预期F1: {nsga2_results['recommended_solution']['objectives']['f1_score']:.4f}")
        
        # 使用推荐组合进行全量训练
        # 注意: 这里需要修改create_bsdcnn_dataloaders来支持患者选择
        # 暂时先输出结果,完整训练流程可以后续实现
        print("\n提示: 请使用推荐的患者组合重新训练模型以获得最终性能")
        print(f"建议命令: python bsdcnn_train.py --selected_patients {' '.join(recommended_patients)} ...")
        
        return
    
    # ========== Standard Training (原有流程) ==========
    # Train model
    model, results = train_model(args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()
