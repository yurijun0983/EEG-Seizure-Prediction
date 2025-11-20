"""
Genetic Algorithm for EEG Channel Selection with BSDCNN Model Integration
集成BSDCNN模型的遗传算法EEG通道选择
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bsdcnn_model import BSDCNN, create_bsdcnn_model


# Standard 29 EEG channels used in Siena Scalp EEG Database
STANDARD_29_CHANNELS = [
    'EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1',
    'EEG F7', 'EEG T3', 'EEG T5',
    'EEG Fc1', 'EEG Fc5', 'EEG Cp1', 'EEG Cp5',
    'EEG F9', 'EEG Fz', 'EEG Pz',
    'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2',
    'EEG F8', 'EEG T4', 'EEG T6',
    'EEG Fc2', 'EEG Fc6', 'EEG Cp2', 'EEG Cp6',
    'EEG F10'
]


class GeneticAlgorithmChannelSelectionWithModel:
    """
    Genetic Algorithm for EEG Channel Selection with BSDCNN Model Integration
    集成BSDCNN模型的遗传算法EEG通道选择
    """
    
    def __init__(self, population_size=30, generations=20, mutation_rate=0.1, 
                 crossover_rate=0.8, elite_size=2, channels=None):
        """
        初始化遗传算法参数
        
        Args:
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_size: 精英个体数量
            channels: 通道列表
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.channels = channels if channels is not None else STANDARD_29_CHANNELS
        self.n_channels = len(self.channels)
    
    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 创建二进制个体，表示通道选择状态
            individual = np.random.randint(0, 2, self.n_channels)
            # 确保至少选择一个通道
            if np.sum(individual) == 0:
                individual[np.random.randint(0, self.n_channels)] = 1
            population.append(individual)
        return population
    
    def _train_and_evaluate_model(self, X_train, y_train, X_val, y_val, num_channels):
        """
        训练和评估BSDCNN模型
        
        Args:
            X_train: 训练数据 (samples, channels, time_points)
            y_train: 训练标签 (samples,)
            X_val: 验证数据 (samples, channels, time_points)
            y_val: 验证标签 (samples,)
            num_channels: 通道数量
            
        Returns:
            f1: F1分数
        """
        try:
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 创建模型
            sequence_length = X_train.shape[2]
            model = BSDCNN(num_channels=num_channels, sequence_length=sequence_length).to(device)
            
            # 转换数据为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.LongTensor(y_train).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            
            # 设置优化器和损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # 训练模型（简化训练以节省时间）
            model.train()
            for epoch in range(5):  # 简化训练，只训练5个epoch
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                f1 = f1_score(y_val, val_predictions, average='weighted')
            
            # 清理GPU内存
            del model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
            torch.cuda.empty_cache()
            
            return f1
            
        except Exception as e:
            print(f"模型训练评估出错: {e}")
            return 0.0
    
    def _evaluate_fitness(self, individual, X, y, n_splits=3):
        """
        使用交叉验证评估个体适应度（完整版）
        集成BSDCNN模型进行准确评估
        
        Args:
            individual: 二进制编码的个体
            X: EEG数据 (samples, channels, time_points)
            y: 标签 (samples,)
            n_splits: 交叉验证折数
            
        Returns:
            fitness: 适应度值（平均F1分数）
        """
        # 获取选中的通道索引
        selected_channels = np.where(individual == 1)[0]
        
        if len(selected_channels) == 0:
            return 0.0
        
        # 选择对应通道的数据
        X_selected = X[:, selected_channels, :]
        
        # 使用交叉验证评估
        f1_scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X_selected, y):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练和评估模型
            f1 = self._train_and_evaluate_model(
                X_train, y_train, X_val, y_val, len(selected_channels)
            )
            f1_scores.append(f1)
        
        # 返回平均F1分数
        return np.mean(f1_scores)
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            # 随机选择几个个体进行锦标赛
            tournament_indices = np.random.choice(len(population), 
                                                size=min(tournament_size, len(population)), 
                                                replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            # 选择适应度最高的个体
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected
    
    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual):
        """位翻转变异"""
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # 翻转位
        # 确保至少选择一个通道
        if np.sum(individual) == 0:
            individual[np.random.randint(0, len(individual))] = 1
        return individual
    
    def _apply_elitism(self, population, fitness_scores):
        """精英保留策略"""
        # 获取适应度最高的精英个体
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elite = [population[i].copy() for i in elite_indices]
        return elite
    
    def get_selected_channels(self, individual):
        """获取选中的通道名称"""
        selected_indices = np.where(individual == 1)[0]
        return [self.channels[i] for i in selected_indices]
    
    def optimize(self, X, y, verbose=True):
        """
        运行遗传算法优化
        
        Args:
            X: EEG数据 (samples, channels, time_points)
            y: 标签 (samples,)
            verbose: 是否打印详细信息
            
        Returns:
            best_individual: 最佳个体
            best_fitness: 最佳适应度
            fitness_history: 适应度历史记录
        """
        # 初始化种群
        population = self._initialize_population()
        fitness_history = []
        
        if verbose:
            print(f"开始遗传算法优化: {self.population_size} 个体 × {self.generations} 代")
            print(f"通道数量: {self.n_channels}")
        
        best_fitness = 0
        best_individual = None
        
        # 迭代优化
        for generation in range(self.generations):
            # 评估种群适应度
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, X, y)
                fitness_scores.append(fitness)
            
            # 记录最佳个体
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            fitness_history.append(best_fitness)
            
            if verbose and (generation + 1) % 5 == 0:
                avg_fitness = np.mean(fitness_scores)
                selected_channels = self.get_selected_channels(best_individual)
                print(f"第 {generation + 1:2d} 代: "
                      f"平均适应度 = {avg_fitness:.4f}, "
                      f"最佳适应度 = {best_fitness:.4f}, "
                      f"选中通道数 = {len(selected_channels)}")
            
            # 精英保留
            elite = self._apply_elitism(population, fitness_scores)
            
            # 选择
            selected_population = self._tournament_selection(population, fitness_scores)
            
            # 交叉和变异生成新种群
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                # 随机选择两个父代
                parent_indices = np.random.choice(len(selected_population), size=2, replace=False)
                parent1 = selected_population[parent_indices[0]]
                parent2 = selected_population[parent_indices[1]]
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小正确
            population = new_population[:self.population_size]
        
        if verbose:
            selected_channels = self.get_selected_channels(best_individual)
            print(f"\n优化完成!")
            print(f"最佳适应度: {best_fitness:.4f}")
            print(f"选中的 {len(selected_channels)} 个通道: {selected_channels}")
        
        return best_individual, best_fitness, fitness_history


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    n_channels = 29
    n_timepoints = 5120
    
    # 模拟EEG数据
    X = np.random.randn(n_samples, n_channels, n_timepoints)
    # 模拟标签（0: interictal, 1: preictal）
    y = np.random.randint(0, 2, n_samples)
    
    # 运行遗传算法
    ga_optimizer = GeneticAlgorithmChannelSelectionWithModel(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    print("运行示例...")
    best_individual, best_fitness, fitness_history = ga_optimizer.optimize(X, y, verbose=True)
    
    print(f"\n示例完成!")
    print(f"最佳适应度: {best_fitness:.4f}")
    selected_channels = ga_optimizer.get_selected_channels(best_individual)
    print(f"选中的通道: {selected_channels}")