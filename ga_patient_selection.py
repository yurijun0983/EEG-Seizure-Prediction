"""
NSGA-II based Multi-Objective Patient Selection for Cross-Patient Seizure Prediction

Based on:
1. "A personalized and evolutionary algorithm for interpretable EEG epilepsy seizure prediction" (Nature, 2021)
2. "Evolutionary transfer optimization-based approach for automated ictal pattern recognition" (Front. Hum. Neurosci., 2024)
3. Research.txt Section V: NSGA-II Multi-Objective Optimization

This module optimizes source patient selection for target patient (PN14) by:
- Objective 1 (Maximize): F1 Score on target patient test/validation set
- Objective 2 (Minimize): Number of selected source patients (improve generalization)
- Objective 3 (Minimize): Training sample imbalance penalty
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from typing import List, Tuple, Dict
from tqdm import tqdm

from bsdcnn_model import create_bsdcnn_model
from bsdcnn_data_loader import create_bsdcnn_dataloaders_with_patient_selection
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class Individual:
    """个体: 表示一个源患者选择方案"""
    def __init__(self, n_patients: int, binary_mode: bool = True):
        """
        Args:
            n_patients: 源患者数量
            binary_mode: True=二进制选择(0/1), False=权重优化(0.0-1.0)
        """
        self.n_patients = n_patients
        self.binary_mode = binary_mode
        
        if binary_mode:
            # 二进制编码: [1,0,1,1,0,...] 表示选择/排除患者
            # 确保至少选择2个患者
            self.genes = np.random.randint(0, 2, n_patients)
            while np.sum(self.genes) < 2:
                self.genes = np.random.randint(0, 2, n_patients)
        else:
            # 权重编码: [0.8,0.2,0.9,...] 表示患者权重
            self.genes = np.random.uniform(0.0, 1.0, n_patients)
        
        # 适应度值 (3个目标)
        self.objectives = np.array([0.0, 0.0, 0.0])  # [f1, n_patients, imbalance]
        
        # Pareto支配相关
        self.rank = 0  # 帕累托层级
        self.crowding_distance = 0.0  # 拥挤距离
        self.dominated_count = 0  # 被支配次数
        self.dominated_set = []  # 该个体支配的个体集合
    
    def get_selected_patients(self, patient_list: List[str]) -> List[str]:
        """获取选中的患者列表"""
        if self.binary_mode:
            return [patient_list[i] for i in range(len(patient_list)) if self.genes[i] == 1]
        else:
            # 权重模式: 选择权重>阈值的患者
            threshold = 0.3
            return [patient_list[i] for i in range(len(patient_list)) if self.genes[i] > threshold]
    
    def get_patient_weights(self) -> np.ndarray:
        """获取患者权重 (仅权重模式使用)"""
        return self.genes if not self.binary_mode else self.genes.astype(float)
    
    def dominates(self, other) -> bool:
        """判断当前个体是否帕累托支配另一个体
        
        支配条件: 在所有目标上不劣于other, 且至少在一个目标上严格优于other
        目标定义:
        - objectives[0]: F1 (越大越好) -> 最小化 -F1
        - objectives[1]: 患者数量 (越小越好) -> 最小化
        - objectives[2]: 样本不平衡度 (越小越好) -> 最小化
        """
        # 转换为最小化问题
        self_obj = np.array([-self.objectives[0], self.objectives[1], self.objectives[2]])
        other_obj = np.array([-other.objectives[0], other.objectives[1], other.objectives[2]])
        
        # 在所有目标上不劣于other
        not_worse = np.all(self_obj <= other_obj)
        # 至少在一个目标上严格优于other
        strictly_better = np.any(self_obj < other_obj)
        
        return not_worse and strictly_better
    
    def __repr__(self):
        selected = np.sum(self.genes > 0.5) if not self.binary_mode else np.sum(self.genes)
        return (f"Individual(selected={selected}/{self.n_patients}, "
                f"F1={self.objectives[0]:.4f}, "
                f"N_patients={self.objectives[1]:.0f}, "
                f"Imbalance={self.objectives[2]:.4f})")


class NSGAII:
    """NSGA-II 多目标遗传算法"""
    
    def __init__(
        self,
        population_size: int = 20,
        n_generations: int = 15,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        tournament_size: int = 2,
        binary_mode: bool = True,
        seed: int = 42
    ):
        """
        Args:
            population_size: 种群大小
            n_generations: 进化代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            tournament_size: 锦标赛选择大小
            binary_mode: True=二进制选择, False=权重优化
            seed: 随机种子
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.binary_mode = binary_mode
        self.seed = seed
        
        np.random.seed(seed)
        
        # 进化历史
        self.history = {
            'best_f1_per_gen': [],
            'avg_f1_per_gen': [],
            'pareto_front_per_gen': [],
            'diversity_per_gen': []
        }
    
    def initialize_population(self, n_patients: int) -> List[Individual]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = Individual(n_patients, self.binary_mode)
            population.append(individual)
        return population
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """锦标赛选择"""
        tournament = np.random.choice(population, self.tournament_size, replace=False)
        
        # 优先选择rank更小的(更优的帕累托前沿)
        tournament = sorted(tournament, key=lambda x: (x.rank, -x.crowding_distance))
        return deepcopy(tournament[0])
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """单点交叉"""
        if np.random.rand() > self.crossover_prob:
            return deepcopy(parent1), deepcopy(parent2)
        
        n = parent1.n_patients
        point = np.random.randint(1, n)
        
        child1 = Individual(n, self.binary_mode)
        child2 = Individual(n, self.binary_mode)
        
        child1.genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2.genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        # 确保二进制模式至少选择2个患者
        if self.binary_mode:
            if np.sum(child1.genes) < 2:
                child1.genes[np.random.randint(0, n)] = 1
                child1.genes[np.random.randint(0, n)] = 1
            if np.sum(child2.genes) < 2:
                child2.genes[np.random.randint(0, n)] = 1
                child2.genes[np.random.randint(0, n)] = 1
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """变异操作"""
        mutant = deepcopy(individual)
        
        for i in range(mutant.n_patients):
            if np.random.rand() < self.mutation_prob:
                if self.binary_mode:
                    # 位翻转
                    mutant.genes[i] = 1 - mutant.genes[i]
                else:
                    # 高斯扰动
                    mutant.genes[i] += np.random.normal(0, 0.1)
                    mutant.genes[i] = np.clip(mutant.genes[i], 0.0, 1.0)
        
        # 确保至少选择2个患者
        if self.binary_mode and np.sum(mutant.genes) < 2:
            mutant.genes[np.random.randint(0, mutant.n_patients)] = 1
            mutant.genes[np.random.randint(0, mutant.n_patients)] = 1
        
        return mutant
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """快速非支配排序 (NSGA-II核心算法)"""
        # 重置支配信息
        for ind in population:
            ind.dominated_count = 0
            ind.dominated_set = []
        
        # 计算支配关系
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                if p.dominates(q):
                    p.dominated_set.append(q)
                elif q.dominates(p):
                    p.dominated_count += 1
        
        # 分层
        fronts = []
        current_front = [ind for ind in population if ind.dominated_count == 0]
        
        for ind in current_front:
            ind.rank = 0
        
        rank = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            for p in current_front:
                for q in p.dominated_set:
                    q.dominated_count -= 1
                    if q.dominated_count == 0:
                        q.rank = rank + 1
                        next_front.append(q)
            
            rank += 1
            current_front = next_front
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # 重置拥挤距离
        for ind in front:
            ind.crowding_distance = 0.0
        
        # 对每个目标维度
        for obj_idx in range(3):
            # 按该目标排序
            front = sorted(front, key=lambda x: x.objectives[obj_idx])
            
            # 边界个体设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算归一化距离
            obj_min = front[0].objectives[obj_idx]
            obj_max = front[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                distance = (front[i + 1].objectives[obj_idx] - 
                           front[i - 1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance
    
    def evolve(self, population: List[Individual]) -> List[Individual]:
        """进化一代"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # 选择
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # 合并父代和子代
        combined = population + offspring[:self.population_size]
        
        # 非支配排序
        fronts = self.fast_non_dominated_sort(combined)
        
        # 选择新种群
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                # 计算拥挤距离
                self.calculate_crowding_distance(front)
                new_population.extend(front)
            else:
                # 最后一层按拥挤距离排序
                self.calculate_crowding_distance(front)
                front = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front[:self.population_size - len(new_population)])
                break
        
        return new_population
    
    def get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """获取帕累托最优前沿 (Rank 0)"""
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    def record_generation(self, generation: int, population: List[Individual]):
        """记录每代统计信息"""
        f1_scores = [ind.objectives[0] for ind in population]
        
        self.history['best_f1_per_gen'].append(max(f1_scores))
        self.history['avg_f1_per_gen'].append(np.mean(f1_scores))
        
        pareto_front = self.get_pareto_front(population)
        self.history['pareto_front_per_gen'].append([
            {
                'genes': ind.genes.tolist(),
                'f1': float(ind.objectives[0]),
                'n_patients': int(ind.objectives[1]),
                'imbalance': float(ind.objectives[2])
            }
            for ind in pareto_front
        ])
        
        # 种群多样性 (基因型多样性)
        gene_matrix = np.array([ind.genes for ind in population])
        diversity = np.mean(np.std(gene_matrix, axis=0))
        self.history['diversity_per_gen'].append(float(diversity))


def evaluate_individual_fitness(
    individual: Individual,
    source_patients: List[str],
    preloaded_data: Dict,  # 新增:预加载的数据字典
    test_patient_data: Dict,  # 新增:测试患者数据
    args,
    device: torch.device,
    use_test_set: bool = False
) -> Tuple[float, float, float]:
    """
    评估个体的适应度 (三个目标)
    
    Args:
        individual: 待评估的个体
        source_patients: 源患者ID列表
        preloaded_data: 预加载的所有源患者数据 {patient_id: {'preictal': [...], 'interictal': [...]}}
        test_patient_data: 测试患者的数据 {'train': [...], 'val': [...], 'test': [...]}
        args: 训练参数
        device: 计算设备
        use_test_set: 是否在测试集上评估 (谨慎使用)
    
    Returns:
        (f1_score, n_selected_patients, imbalance_penalty)
    """
    from torch.utils.data import DataLoader
    from bsdcnn_data_loader import BSDCNNDataset
    from sklearn.model_selection import train_test_split
    
    # 获取选中的患者
    selected_patients = individual.get_selected_patients(source_patients)
    n_selected = len(selected_patients)
    
    # 目标2: 患者数量 (越少越好,提高泛化性)
    n_patients_penalty = n_selected
    
    # 从预加载数据中收集选中患者的数据
    all_preictal = []
    all_interictal = []
    
    for patient_id in selected_patients:
        if patient_id in preloaded_data:
            patient_data = preloaded_data[patient_id]
            preictal = patient_data['preictal']
            interictal = patient_data['interictal']
            
            # 应用患者权重 (如果使用权重模式)
            if not individual.binary_mode:
                patient_idx = source_patients.index(patient_id)
                weight = individual.genes[patient_idx]
                
                if weight < 1.0:
                    n_preictal = max(1, int(len(preictal) * weight))
                    n_interictal = max(1, int(len(interictal) * weight))
                    preictal = preictal[:n_preictal]
                    interictal = interictal[:n_interictal]
            
            all_preictal.extend(preictal)
            all_interictal.extend(interictal)
    
    # 目标3: 样本不平衡惩罚
    if len(all_preictal) > 0 and len(all_interictal) > 0:
        imbalance_ratio = max(len(all_preictal), len(all_interictal)) / (min(len(all_preictal), len(all_interictal)) + 1e-6)
    else:
        imbalance_ratio = 10.0
    imbalance_penalty = np.log(imbalance_ratio + 1)
    
    # 如果数据太少,返回低适应度
    if len(all_preictal) < 10 or len(all_interictal) < 10:
        return 0.0, n_patients_penalty, imbalance_penalty
    
    # 划分训练/验证集
    train_preictal, val_preictal = train_test_split(
        all_preictal, test_size=args.val_split, random_state=42, shuffle=True
    )
    train_interictal, val_interictal = train_test_split(
        all_interictal, test_size=args.val_split, random_state=42, shuffle=True
    )
    
    # 创建数据集
    train_segments = train_preictal + train_interictal
    train_labels = [1] * len(train_preictal) + [0] * len(train_interictal)
    train_dataset = BSDCNNDataset(train_segments, train_labels, augment=True)
    
    val_segments = val_preictal + val_interictal
    val_labels = [1] * len(val_preictal) + [0] * len(val_interictal)
    val_dataset = BSDCNNDataset(val_segments, val_labels, augment=False)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # 测试集DataLoader
    if use_test_set:
        test_dataset = BSDCNNDataset(
            test_patient_data['test_segments'],
            test_patient_data['test_labels'],
            augment=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        eval_loader = test_loader
    else:
        eval_loader = val_loader
    
    # 获取输入维度
    sample_batch, _ = next(iter(train_loader))
    num_channels = sample_batch.shape[1]
    sequence_length = sample_batch.shape[2]
    
    # 创建模型 (根据模型类型使用正确的参数)
    model_type = getattr(args, 'model_type', 'standard')  # 默认使用standard模型
    model_kwargs = {
        'num_channels': num_channels,
        'sequence_length': sequence_length,
        'num_classes': 2
    }
    
    # 只有BSDCNN模型支持use_binary_activation参数
    if model_type in ['full', 'simplified']:
        model_kwargs['use_binary_activation'] = getattr(args, 'use_binary_activation', False)
    
    model = create_bsdcnn_model(model_type, **model_kwargs)
    model = model.to(device)
    
    # 损失函数
    class_counts = [len(train_interictal), len(train_preictal)]
    if args.use_weighted_loss and class_counts[0] > 0 and class_counts[1] > 0:
        total = sum(class_counts)
        weights = torch.FloatTensor([total / (2 * class_counts[0]), 
                                     total / (2 * class_counts[1])]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 快速训练 (仅训练少量epoch评估适应度)
    n_epochs = args.ga_train_epochs
    
    for epoch in range(n_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in eval_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    # 目标1: F1 Score (越大越好)
    if len(all_labels) > 0:
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    else:
        f1 = 0.0
    
    # 清理
    del model, optimizer, train_loader, val_loader
    if use_test_set:
        del test_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return f1, n_patients_penalty, imbalance_penalty


def run_nsga2_patient_selection(
    source_patients: List[str],
    args,
    device: torch.device,
    output_dir: str
) -> Dict:
    """
    运行NSGA-II患者选择优化
    
    Args:
        source_patients: 源患者ID列表 (例如 ['PN00','PN03',...,'PN17'])
        args: 训练参数
        device: 计算设备
        output_dir: 输出目录
    
    Returns:
        包含帕累托前沿和最优个体的字典
    """
    print("\n" + "="*70)
    print("NSGA-II Multi-Objective Patient Selection Optimization")
    print("="*70)
    print(f"Source Patients: {source_patients}")
    print(f"Target Patient: {args.test_patient}")
    print(f"Population Size: {args.ga_ps_population}")
    print(f"Generations: {args.ga_ps_generations}")
    print(f"Training Epochs per Evaluation: {args.ga_train_epochs}")
    print(f"Evaluation Dataset: {'Test Set' if args.ga_ps_use_test else 'Validation Set'}")
    print("="*70 + "\n")
    
    # ====== 新增: 预加载所有源患者数据 ======
    print("\n" + "="*70)
    print("✨ Preloading All Source Patient Data (One-time Cost)")
    print("="*70)
    
    preloaded_data = {}
    
    for patient_id in source_patients:
        print(f"Loading {patient_id}...", end=" ")
        
        # 从预处理文件加载
        if args.preprocessed_dir and os.path.exists(os.path.join(args.preprocessed_dir, f'{patient_id}.npz')):
            npz_path = os.path.join(args.preprocessed_dir, f'{patient_id}.npz')
            data = np.load(npz_path)
            preictal = [np.array(seg, dtype=np.float32) for seg in data['preictal']]
            interictal = [np.array(seg, dtype=np.float32) for seg in data['interictal']]
            data.close()
        else:
            # 如果没有预处理文件,需要从原始数据加载 (较慢)
            from bsdcnn_data_loader import load_patient_data
            preictal, interictal = load_patient_data(
                args.data_root, patient_id,
                window_seconds=getattr(args, 'window_seconds', 5),
                overlap_seconds=getattr(args, 'overlap_seconds', 2.5)
            )
            preictal = [np.array(seg, dtype=np.float32) for seg in preictal]
            interictal = [np.array(seg, dtype=np.float32) for seg in interictal]
        
        preloaded_data[patient_id] = {
            'preictal': preictal,
            'interictal': interictal
        }
        
        print(f"✅ {len(preictal)} preictal, {len(interictal)} interictal")
    
    print(f"\n✨ All source patient data preloaded! Total patients: {len(preloaded_data)}")
    print("="*70 + "\n")
    
    # ====== 预加载测试患者数据 ======
    print(f"Loading test patient data ({args.test_patient})...", end=" ")
    
    if args.preprocessed_dir and os.path.exists(os.path.join(args.preprocessed_dir, f'{args.test_patient}.npz')):
        npz_path = os.path.join(args.preprocessed_dir, f'{args.test_patient}.npz')
        data = np.load(npz_path)
        test_preictal = [np.array(seg, dtype=np.float32) for seg in data['preictal']]
        test_interictal = [np.array(seg, dtype=np.float32) for seg in data['interictal']]
        data.close()
    else:
        from bsdcnn_data_loader import load_patient_data
        test_preictal, test_interictal = load_patient_data(
            args.data_root, args.test_patient,
            window_seconds=getattr(args, 'window_seconds', 5),
            overlap_seconds=getattr(args, 'overlap_seconds', 2.5)
        )
        test_preictal = [np.array(seg, dtype=np.float32) for seg in test_preictal]
        test_interictal = [np.array(seg, dtype=np.float32) for seg in test_interictal]
    
    # 合并测试数据
    test_segments = test_preictal + test_interictal
    test_labels = [1] * len(test_preictal) + [0] * len(test_interictal)
    
    test_patient_data = {
        'test_segments': test_segments,
        'test_labels': test_labels
    }
    
    print(f"✅ {len(test_preictal)} preictal, {len(test_interictal)} interictal")
    print("\n" + "="*70 + "\n")
    
    # 初始NSGA-II
    nsga2 = NSGAII(
        population_size=args.ga_ps_population,
        n_generations=args.ga_ps_generations,
        crossover_prob=0.9,
        mutation_prob=0.2,
        tournament_size=2,
        binary_mode=(args.ga_ps_mode == 'binary'),
        seed=args.seed
    )
    
    # 初始化种群
    print("Initializing population...")
    population = nsga2.initialize_population(len(source_patients))
    
    # 进化循环
    for generation in range(args.ga_ps_generations):
        print(f"\n{'='*70}")
        print(f"Generation {generation + 1}/{args.ga_ps_generations}")
        print(f"{'='*70}")
        
        # 评估种群 (使用预加载的数据)
        print("Evaluating population fitness...")
        pbar = tqdm(population, desc=f"Gen {generation + 1}")
        
        for individual in pbar:
            f1, n_patients, imbalance = evaluate_individual_fitness(
                individual=individual,
                source_patients=source_patients,
                preloaded_data=preloaded_data,  # 使用预加载数据!
                test_patient_data=test_patient_data,
                args=args,
                device=device,
                use_test_set=args.ga_ps_use_test
            )
            
            individual.objectives = np.array([f1, n_patients, imbalance])
            
            pbar.set_postfix({
                'F1': f'{f1:.4f}',
                'N_patients': int(n_patients),
                'Imbalance': f'{imbalance:.3f}'
            })
        
        # 记录当前代统计
        nsga2.record_generation(generation, population)
        
        # 获取当前帕累托前沿
        pareto_front = nsga2.get_pareto_front(population)
        
        print(f"\nGeneration {generation + 1} Summary:")
        print(f"  Best F1: {nsga2.history['best_f1_per_gen'][-1]:.4f}")
        print(f"  Avg F1: {nsga2.history['avg_f1_per_gen'][-1]:.4f}")
        print(f"  Pareto Front Size: {len(pareto_front)}")
        print(f"  Population Diversity: {nsga2.history['diversity_per_gen'][-1]:.4f}")
        
        # 显示部分帕累托前沿个体
        print(f"\n  Top Pareto Front Individuals:")
        for i, ind in enumerate(pareto_front[:5]):
            selected = ind.get_selected_patients(source_patients)
            print(f"    #{i+1}: F1={ind.objectives[0]:.4f}, "
                  f"N={int(ind.objectives[1])}, "
                  f"Imb={ind.objectives[2]:.3f}, "
                  f"Patients={selected}")
        
        # 进化到下一代 (最后一代不进化)
        if generation < args.ga_ps_generations - 1:
            print("\nEvolving to next generation...")
            population = nsga2.evolve(population)
    
    # 最终帕累托前沿
    print(f"\n{'='*70}")
    print("Optimization Complete!")
    print(f"{'='*70}")
    
    final_pareto_front = nsga2.get_pareto_front(population)
    
    print(f"\nFinal Pareto Front ({len(final_pareto_front)} solutions):")
    for i, ind in enumerate(final_pareto_front):
        selected = ind.get_selected_patients(source_patients)
        print(f"  Solution #{i+1}:")
        print(f"    F1 Score: {ind.objectives[0]:.4f}")
        print(f"    N Patients: {int(ind.objectives[1])}")
        print(f"    Imbalance: {ind.objectives[2]:.3f}")
        print(f"    Selected Patients: {selected}")
        print()
    
    # 选择推荐方案 (基于加权适应度)
    # 根据memory: α×PrimaryMetric - β×Complexity - γ×Cost
    best_individual = None
    best_weighted_fitness = -float('inf')
    
    alpha, beta, gamma = 10.0, 0.5, 0.3  # 权重参数
    
    for ind in final_pareto_front:
        weighted_fitness = (alpha * ind.objectives[0] - 
                           beta * ind.objectives[1] - 
                           gamma * ind.objectives[2])
        if weighted_fitness > best_weighted_fitness:
            best_weighted_fitness = weighted_fitness
            best_individual = ind
    
    best_selected = best_individual.get_selected_patients(source_patients)
    
    print(f"{'='*70}")
    print("Recommended Solution (Weighted Fitness):")
    print(f"  F1 Score: {best_individual.objectives[0]:.4f}")
    print(f"  N Patients: {int(best_individual.objectives[1])}")
    print(f"  Imbalance: {best_individual.objectives[2]:.3f}")
    print(f"  Selected Patients: {best_selected}")
    print(f"  Weighted Fitness: {best_weighted_fitness:.4f}")
    print(f"{'='*70}\n")
    
    # 保存结果
    results = {
        'algorithm': 'NSGA-II',
        'source_patients': source_patients,
        'target_patient': args.test_patient,
        'parameters': {
            'population_size': args.ga_ps_population,
            'n_generations': args.ga_ps_generations,
            'crossover_prob': 0.9,
            'mutation_prob': 0.2,
            'mode': args.ga_ps_mode
        },
        'history': nsga2.history,
        'pareto_front': [
            {
                'genes': ind.genes.tolist(),
                'selected_patients': ind.get_selected_patients(source_patients),
                'objectives': {
                    'f1_score': float(ind.objectives[0]),
                    'n_patients': int(ind.objectives[1]),
                    'imbalance': float(ind.objectives[2])
                }
            }
            for ind in final_pareto_front
        ],
        'recommended_solution': {
            'genes': best_individual.genes.tolist(),
            'selected_patients': best_selected,
            'objectives': {
                'f1_score': float(best_individual.objectives[0]),
                'n_patients': int(best_individual.objectives[1]),
                'imbalance': float(best_individual.objectives[2])
            },
            'weighted_fitness': float(best_weighted_fitness)
        }
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到JSON
    results_path = os.path.join(output_dir, 'nsga2_patient_selection_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {results_path}\n")
    
    return results
