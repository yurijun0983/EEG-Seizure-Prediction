"""
EEG-specific data augmentation techniques for seizure prediction
Based on recent literature review
"""
import numpy as np
import torch


class EEGAugmentation:
    """
    Data augmentation techniques for EEG signals
    """
    
    @staticmethod
    def add_gaussian_noise(x, std=0.01):
        """
        Add Gaussian noise to simulate measurement noise
        
        Args:
            x: EEG signal (channels, samples)
            std: Standard deviation of noise
        """
        noise = np.random.normal(0, std, x.shape)
        return x + noise
    
    @staticmethod
    def time_shift(x, max_shift=50):
        """
        Random time shift augmentation
        
        Args:
            x: EEG signal (channels, samples)
            max_shift: Maximum shift in samples
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(x, shift, axis=1)
    
    @staticmethod
    def amplitude_scale(x, scale_range=(0.9, 1.1)):
        """
        Random amplitude scaling
        
        Args:
            x: EEG signal (channels, samples)
            scale_range: (min, max) scaling factors
        """
        scale = np.random.uniform(*scale_range)
        return x * scale
    
    @staticmethod
    def channel_dropout(x, dropout_prob=0.1):
        """
        Randomly zero out channels
        
        Args:
            x: EEG signal (channels, samples)
            dropout_prob: Probability of dropping each channel
        """
        mask = np.random.binomial(1, 1-dropout_prob, size=x.shape[0])
        return x * mask[:, np.newaxis]
    
    @staticmethod
    def time_warp(x, sigma=0.2, num_knots=4):
        """
        Time warping augmentation
        
        Args:
            x: EEG signal (channels, samples)
            sigma: Warping strength
            num_knots: Number of warping knots
        """
        n_channels, n_samples = x.shape
        warped = np.zeros_like(x)
        
        # Create warping curve
        orig_steps = np.linspace(0, n_samples-1, num_knots)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=num_knots)
        warp_steps = np.cumsum(orig_steps * random_warps)
        warp_steps = warp_steps * (n_samples-1) / warp_steps[-1]
        
        # Interpolate
        for ch in range(n_channels):
            warped[ch] = np.interp(np.arange(n_samples), warp_steps, x[ch])
        
        return warped
    
    @staticmethod
    def mixup(x1, x2, alpha=0.2):
        """
        Mixup augmentation between two samples
        
        Args:
            x1, x2: Two EEG signals
            alpha: Mixup ratio
        """
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2
    
    def __call__(self, x, augment_prob=0.5):
        """
        Apply random augmentation
        
        Args:
            x: EEG signal (channels, samples)
            augment_prob: Probability of applying each augmentation
        """
        x_aug = x.copy()
        
        if np.random.rand() < augment_prob:
            x_aug = self.add_gaussian_noise(x_aug, std=0.01)
        
        if np.random.rand() < augment_prob:
            x_aug = self.time_shift(x_aug, max_shift=50)
        
        if np.random.rand() < augment_prob:
            x_aug = self.amplitude_scale(x_aug, scale_range=(0.95, 1.05))
        
        if np.random.rand() < 0.1:  # Lower probability for channel dropout
            x_aug = self.channel_dropout(x_aug, dropout_prob=0.1)
        
        return x_aug


class GACrossoverAugmentor:
    """
    基于遗传算法个体参数的EEG交叉增强器
    用于将发作间期(interictal)和发作期/发作前期(ictal/preictal)片段进行交叉混合生成新样本
    
    个体参数示例:
        params = {
            'type': 'time' | 'channel' | 'window',  # 交叉类型
            'lambda': 0.3,              # 混合比例 [0, 1]
            'pos_only': True,           # 仅对正类(preictal)做增强
            'intra_patient': True,      # 仅同患者配对
            'time_mask_density': 0.2    # 对time型交叉的掩蔽密度
        }
    
    参考文献思想:
    - Mixup数据增强策略 (Zhang et al., 2018)
    - 遗传算法用于EEG特征优化 (Frontiers Neuroscience, 2023)
    - 时序数据的交叉验证增强 (IEEE Sensors, 2022)
    """
    
    def __init__(self, params):
        """
        初始化交叉增强器
        
        Args:
            params: dict, 包含交叉策略的参数配置
        """
        self.params = params
    
    def crossover_pair(self, x_pos, x_neg):
        """
        对一对EEG片段（发作前期与发作间期）执行交叉操作
        
        Args:
            x_pos: 发作前期片段 (channels, samples)
            x_neg: 发作间期片段 (channels, samples)
        
        Returns:
            交叉后的新片段 (channels, samples)
        """
        crossover_type = self.params.get('type', 'window')
        lam = float(self.params.get('lambda', 0.5))
        lam = max(0.0, min(1.0, lam))  # 限制在 [0, 1]
        
        if crossover_type == 'window':
            # 整窗口线性混合（类似mixup）
            # 参考: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
            return lam * x_pos + (1.0 - lam) * x_neg
        
        elif crossover_type == 'channel':
            # 按通道选择混合：一部分通道来自正类，剩余来自负类
            # 参考: 遗传算法的单点交叉思想应用于通道维度
            C, T = x_pos.shape
            c_split = np.random.randint(1, C)
            idx = np.random.permutation(C)
            pos_idx = idx[:c_split]
            neg_idx = idx[c_split:]
            out = np.zeros_like(x_pos)
            out[pos_idx] = lam * x_pos[pos_idx] + (1.0 - lam) * x_neg[pos_idx]
            out[neg_idx] = lam * x_neg[neg_idx] + (1.0 - lam) * x_pos[neg_idx]
            return out
        
        elif crossover_type == 'time':
            # 按时间片交叉：随机掩蔽一部分时间片，来自正类，其余来自负类
            # 参考: SpecAugment思想用于时序信号掩蔽与混合
            C, T = x_pos.shape
            density = float(self.params.get('time_mask_density', 0.2))
            density = max(0.0, min(1.0, density))
            mask = (np.random.rand(T) < density).astype(np.float32)
            mask = np.tile(mask, (C, 1))
            return lam * (x_pos * mask + x_neg * (1 - mask)) + (1.0 - lam) * (x_neg * mask + x_pos * (1 - mask))
        
        else:
            # 默认整窗口混合
            return lam * x_pos + (1.0 - lam) * x_neg


def ga_generate_augmented_batch(positives, negatives, params, batch_size):
    """
    用GA个体参数生成一个增强批次
    通过交叉发作间期和发作前期片段生成新的训练样本
    
    Args:
        positives: list[np.ndarray], 发作前期片段列表，每个形状 (channels, samples)
        negatives: list[np.ndarray], 发作间期片段列表，每个形状 (channels, samples)
        params: dict, GA个体参数（交叉类型、混合比例等）
        batch_size: int, 生成的增强样本数量
    
    Returns:
        augmented_segments: list[np.ndarray], 增强后的片段列表
        labels: list[int], 对应的标签（通常标记为正类以强化决策边界）
    """
    augmentor = GACrossoverAugmentor(params)
    augmented_segments = []
    labels = []
    
    for _ in range(batch_size):
        # 随机采样一个正类与一个负类片段进行交叉
        x_pos = positives[np.random.randint(0, len(positives))]
        x_neg = negatives[np.random.randint(0, len(negatives))]
        x_new = augmentor.crossover_pair(x_pos, x_neg)
        
        # 是否仅增强正类
        pos_only = bool(params.get('pos_only', True))
        if pos_only:
            # 交叉样本标记为正类，用于增强正类样本多样性
            augmented_segments.append(x_new.astype(np.float32))
            labels.append(1)
        else:
            # 可选：生成对称增强样本（正负类均衡）
            augmented_segments.append(x_new.astype(np.float32))
            labels.append(np.random.choice([0, 1], p=[0.5, 0.5]))
    
    return augmented_segments, labels
