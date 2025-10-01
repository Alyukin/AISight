import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from typing import Tuple


class CTAnomalyDetector(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 128),
                 latent_dim: int = 512, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Энкодер на основе 3D ResNet
        self.encoder = ResNet(
            block='basic',
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=1,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type='B',
            spatial_dims=3,
            num_classes=2
        )
        

        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            nn.ReLU(inplace=True), 
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
            # НЕ включаем avgpool и fc!
        )
        
        # Автоэнкодер для реконструкции
        self.decoder = self._build_decoder()
        
        # Классификатор аномалий
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Регион предложений для локализации
        self.region_proposal = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 6, 1)
        )
    
    def _build_decoder(self) -> nn.Module:
        """Построение декодера для восстановления до 128x128x128"""
        return nn.Sequential(
            # Input: [4, 512, 4, 4, 4]
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),  # [4, 256, 8, 8, 8]
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # [4, 128, 16, 16, 16]
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),   # [4, 64, 32, 32, 32]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),    # [4, 32, 64, 64, 64]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),     # [4, 1, 128, 128, 128]
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Кодирование через convolutional layers только
        features = self.encoder_layers(x)
        
        # Реконструкция
        reconstructed = self.decoder(features)
        
        # Классификация аномалий
        anomaly_score = self.classifier(features)
        
        # Локализация аномалий
        bbox = self.region_proposal(features)
        
        return anomaly_score, reconstructed, bbox

class AnomalyLoss(nn.Module):
    """Функция потерь для обнаружения аномалий"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.BCELoss()
    
    def forward(self, predictions, batch_data):
        anomaly_pred, reconstructed, _ = predictions
        input_images = batch_data['image']
        anomaly_true = batch_data['label']
        
        # Убедимся, что метки имеют правильный тип и диапазон
        anomaly_true = anomaly_true.float()  # Конвертируем в float
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, input_images)
        
        # Classification loss - ИСПРАВЛЕНИЕ РАЗМЕРОВ
        print(f"anomaly_pred shape: {anomaly_pred.shape}")  # Должно быть [batch_size, 1]
        print(f"anomaly_true shape: {anomaly_true.shape}")  # Должно быть [batch_size]
        
        # Сжимаем anomaly_pred до [batch_size] или расширяем anomaly_true до [batch_size, 1]
        if anomaly_pred.dim() == 2 and anomaly_pred.shape[1] == 1:
            anomaly_pred = anomaly_pred.squeeze(1)  # [batch_size, 1] -> [batch_size]
        
        class_loss = self.classification_loss(anomaly_pred, anomaly_true)
        
        total_loss = self.alpha * recon_loss + self.beta * class_loss
        
        print(f"Losses - recon: {recon_loss:.4f}, class: {class_loss:.4f}, total: {total_loss:.4f}")
        
        return total_loss