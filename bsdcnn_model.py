"""
Binary Single-Dimensional Convolutional Neural Network (BSDCNN) for Seizure Prediction
Paper: Binary Single-Dimensional Convolutional Neural Network for Seizure Prediction

This implementation includes:
- Binary weight convolutions (weights are +1 or -1)
- Binary activations using sign function
- 1D convolutional layers for EEG signal processing
- Lightweight architecture suitable for hardware deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryActivation(nn.Module):
    """
    Binary activation function using sign function.
    During forward pass: output = sign(input)
    During backward pass: use straight-through estimator (STE)
    """
    def __init__(self):
        super(BinaryActivation, self).__init__()
    
    def forward(self, x):
        # Forward pass: binarize to {-1, +1}
        binary_output = torch.sign(x)
        # Replace zeros with +1
        binary_output[binary_output == 0] = 1
        
        # Backward pass: use straight-through estimator
        # Gradient flows through as if identity function
        return binary_output + x - x.detach()


class BinaryConv1d(nn.Module):
    """
    1D Convolutional layer with binary weights.
    Weights are constrained to {-1, +1} values.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BinaryConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize full-precision weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def binarize_weights(self):
        """Binarize weights to {-1, +1}"""
        binary_weights = torch.sign(self.weight)
        # Replace zeros with +1
        binary_weights[binary_weights == 0] = 1
        # Use straight-through estimator for gradients
        return binary_weights + self.weight - self.weight.detach()
    
    def forward(self, x):
        # Binarize weights during forward pass
        binary_w = self.binarize_weights()
        return F.conv1d(x, binary_w, self.bias, self.stride, self.padding)


class BSDCNN(nn.Module):
    """
    Binary Single-Dimensional Convolutional Neural Network for Seizure Prediction.
    
    Architecture:
    - Input: Raw EEG signals (batch_size, num_channels, sequence_length)
    - Multiple binary 1D convolutional layers
    - Binary activations
    - Batch normalization for training stability
    - Global average pooling
    - Fully connected classifier
    
    Args:
        num_channels: Number of EEG channels (e.g., 21 for standard 10-20 system)
        sequence_length: Length of input sequence (e.g., 5120 for 10s at 512Hz)
        num_classes: Number of output classes (2 for binary classification)
        conv_channels: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        use_binary_activation: Whether to use binary activation (default: True)
    """
    def __init__(self, 
                 num_channels=21, 
                 sequence_length=5120,
                 num_classes=2,
                 conv_channels=None,  # Auto-adjust based on sequence length
                 kernel_sizes=None,   # Auto-adjust based on sequence length
                 use_binary_activation=True):
        super(BSDCNN, self).__init__()
        
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_binary_activation = use_binary_activation
        
        # Auto-adjust architecture based on sequence length
        if conv_channels is None or kernel_sizes is None:
            if sequence_length >= 5000:  # 10-second window
                conv_channels = [32, 64, 128, 256]
                kernel_sizes = [64, 32, 16, 8]
            else:  # 5-second window (2560 samples)
                conv_channels = [32, 64, 128, 256]
                kernel_sizes = [32, 16, 8, 4]  # Smaller kernels for shorter signals
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        in_ch = num_channels
        for i, (out_ch, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            # Binary convolution
            self.conv_layers.append(
                BinaryConv1d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size//2)
            )
            # Batch normalization for training stability
            self.bn_layers.append(nn.BatchNorm1d(out_ch))
            # Binary or ReLU activation
            if use_binary_activation:
                self.activation_layers.append(BinaryActivation())
            else:
                self.activation_layers.append(nn.ReLU())
            
            in_ch = out_ch
        
        # Calculate output size after convolutions
        self.feature_size = conv_channels[-1]
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, sequence_length)
        
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        for conv, bn, activation in zip(self.conv_layers, self.bn_layers, self.activation_layers):
            x = conv(x)
            x = bn(x)
            x = activation(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_binary_ops(self):
        """Estimate number of binary operations"""
        total_ops = 0
        input_size = self.sequence_length
        
        for conv in self.conv_layers:
            kernel_ops = conv.kernel_size * conv.in_channels * conv.out_channels
            output_size = (input_size + 2 * conv.padding - conv.kernel_size) // conv.stride + 1
            total_ops += kernel_ops * output_size
            input_size = output_size
        
        return total_ops


class BSDCNNSimplified(nn.Module):
    """
    Simplified version of BSDCNN with fewer layers for faster training.
    Suitable for quick experimentation and testing.
    """
    def __init__(self, num_channels=21, sequence_length=5120, num_classes=2):
        super(BSDCNNSimplified, self).__init__()
        
        # Simplified architecture with 3 conv layers
        self.conv1 = BinaryConv1d(num_channels, 32, kernel_size=64, stride=4, padding=32)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = BinaryActivation()
        
        self.conv2 = BinaryConv1d(32, 64, kernel_size=32, stride=4, padding=16)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = BinaryActivation()
        
        self.conv3 = BinaryConv1d(64, 128, kernel_size=16, stride=4, padding=8)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = BinaryActivation()
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class StandardCNN(nn.Module):
    """
    Standard (non-binary) CNN for comparison/debugging.
    Uses regular Conv1d instead of BinaryConv1d.
    """
    def __init__(self, num_channels=27, sequence_length=5120, num_classes=2):
        super(StandardCNN, self).__init__()
        
        # Standard convolutions
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=64, stride=2, padding=32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AttentionCNNBiLSTM(nn.Module):
    """
    结合注意力机制的CNN-BiLSTM模型
    """
    def __init__(self, num_channels=21, sequence_length=5120, num_classes=2):
        super(AttentionCNNBiLSTM, self).__init__()
        
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        
        # CNN特征提取层
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=64, stride=2, padding=32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8)
        self.bn3 = nn.BatchNorm1d(128)
        
        # BiLSTM层
        self.lstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        
        # 分类器
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 调整维度以适应LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # BiLSTM
        x, _ = self.lstm(x)
        
        # 注意力机制
        attn_output, _ = self.attention(x, x, x)
        
        # 全局平均池化
        x = torch.mean(attn_output, dim=1)
        
        # 分类器
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class TransformerEEG(nn.Module):
    """
    基于Transformer的EEG信号分类模型
    """
    def __init__(self, num_channels=21, sequence_length=5120, num_classes=2, d_model=128, nhead=8, num_layers=3):
        super(TransformerEEG, self).__init__()
        
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(num_channels, d_model)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(sequence_length, d_model))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类器
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 调整输入维度 (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.positional_encoding[:x.size(1), :]
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类器
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_bsdcnn_model(model_type='full', **kwargs):
    """
    Factory function to create BSDCNN model
    
    Args:
        model_type: 'full', 'simplified', 'standard', 'attention_cnn_bilstm', or 'transformer'
        **kwargs: Additional arguments passed to the model
    
    Returns:
        model: BSDCNN model instance
    """
    if model_type == 'full':
        return BSDCNN(**kwargs)
    elif model_type == 'simplified':
        return BSDCNNSimplified(**kwargs)
    elif model_type == 'standard':
        return StandardCNN(**kwargs)
    elif model_type == 'attention_cnn_bilstm':
        return AttentionCNNBiLSTM(**kwargs)
    elif model_type == 'transformer':
        return TransformerEEG(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    print("Testing BSDCNN Model")
    print("=" * 50)
    
    # Create model
    model = BSDCNN(num_channels=21, sequence_length=5120, num_classes=2)
    
    # Print model info
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Binary operations: {model.get_binary_ops():,}")
    print()
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 21, 5120)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print()
    
    # Test simplified model
    print("Testing Simplified BSDCNN Model")
    print("=" * 50)
    model_simple = BSDCNNSimplified(num_channels=21, sequence_length=5120, num_classes=2)
    print(f"Simplified model parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
    output_simple = model_simple(x)
    print(f"Output shape: {output_simple.shape}")
