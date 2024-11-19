import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class ConvEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, fc_dropout_rate=0.1):
        super(ConvEmbedder, self).__init__()
        
        # Configurations
        self.embedding_dim = embedding_dim
        
        self.conv_layers = nn.Sequential(
            #TODO: Check kernel and padding
            nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # Pooling Layer
        self.global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.embedding_layer = nn.Linear(512, embedding_dim)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

    def forward(self, x):
        print(f"ConvEmbedder input shape: {x.shape}")
        
        # Reshape to (batch_size, feature_channels, num_steps, height, width) for 3D Conv
        x = x.permute(0, 2, 1, 3, 4)
        print(f"ConvEmbedder after permute shape: {x.shape}")

        # Apply 3D Convolutions
        x = self.conv_layers(x)
        print(f"ConvEmbedder after conv_layers shape: {x.shape}")
        
        # Apply Global Pooling
        x = self.global_pooling(x)  # Output shape: (batch_size, channels, 1, 1, 1)
        print(f"ConvEmbedder after pooling shape: {x.shape}")
        x = x.view(x.size(0), -1)   # Flatten to (batch_size, channels)
        print(f"ConvEmbedder after flatten shape: {x.shape}")
        
        # Fully Connected Layers
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.embedding_layer(x)
        
        # TODO: l2_normalize - Ensure the weight_decay parameter is set in the optimizer
        
        return x



class BaseModel(nn.Module):
    def __init__(self):
        """
        Args:
            pretrained (bool): Whether to use pretrained weights.
            target_layer (str): The ResNet layer to extract features from ('layer4' corresponds to conv4_x).
        """
        super(BaseModel, self).__init__()

        # Load ResNet-50
        resnet = models.resnet50(pretrained=True)
        layers = list(resnet.children())

        # Stop at layer3 (conv4_x)
        self.base_model = nn.Sequential(*layers[:7])

        # only training BatchNorm layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        print(f"BaseModel input shape: {x.shape}")
        
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        print(f"BaseModel after reshape shape: {x.shape}")

        # Extract features
        features = self.base_model(x)
        print(f"BaseModel after base_model shape: {features.shape}")

        # Restore temporal dimension
        feature_channels, h, w = features.shape[1:]
        features = features.view(batch_size, num_frames, feature_channels, h, w)
        print(f"BaseModel output shape: {features.shape}")

        return features
    
class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = BaseModel()
        self.emb = ConvEmbedder()
        
        


def test_baseModel():
    dummy_input = torch.randn(4, 9, 3, 224, 224)  # 4 videos, 10 frames each
    
    model = BaseModel()
    output = model(dummy_input)
    
    print("Output shape:", output.shape)  # Expected: (4, 10, 1024, 14, 14)

def test_convEmbedder():
    dummy_input = torch.randn(4, 3, 1024, 14, 14)
    
    model = ConvEmbedder()
    output = model(dummy_input)
    
    print("Output shape:", output.shape)  # Expected: (4, 128)

def test_algorithm():
    dummy_data = {
    'frames': torch.randn(4, 9, 3, 224, 224)
}
    model = Algorithm()
    output = model(dummy_data)
    print("Output shape:", output.shape)  # Expected: [batch_size, num_frames//3, embedding_dim]

if __name__ == "__main__":
    # test_baseModel()
    # test_convEmbedder()
    test_algorithm()
