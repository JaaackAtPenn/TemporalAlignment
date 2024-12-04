import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights



class ConvEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, fc_dropout_rate=0.1, dont_stack=False, small_embedder=False):
        super(ConvEmbedder, self).__init__()
        
        # Configurations
        self.embedding_dim = embedding_dim
        channels = 256 if small_embedder else 512

        self.conv_layers = nn.Sequential(
            #TODO: Check kernel and padding
            nn.Conv3d(in_channels=1024, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        
        # Pooling Layer
        self.global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1)) if not dont_stack else nn.AdaptiveAvgPool2d((1, 1))
        self.dont_stack = dont_stack 
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.embedding_layer = nn.Linear(channels, embedding_dim)
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

    def forward(self, x):
        
        # Reshape to (batch_size, feature_channels, num_steps, height, width) for 3D Conv
        x = x.permute(0, 2, 1, 3, 4)

        # Apply 3D Convolutions
        x = self.conv_layers(x)
        
        # Apply Global Pooling
        x = self.global_pooling(x)  # Output shape: (batch_size, channels, 1, 1, 1) if not dont_stack else (batch_size, channels, steps, 1, 1)
        if self.dont_stack:
            x = x.permute(0, 2, 1, 3, 4)
            steps = x.shape[1]
            batch_size = x.shape[0]
            x = x.contiguous().view(batch_size * steps, -1)       # Reshape to (batch_size * steps, channels), all frames are kept
        else:
            x = x.view(x.size(0), -1)
        
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
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # print("ResNet-50: Loaded")
        # print(resnet)
        layers = list(resnet.children())
        layers[6] = nn.Sequential(*list(layers[6])[:3])

        # Stop at layer3 (conv4_x), conv4c actually
        self.base_model = nn.Sequential(*layers[:7])
        # print("ResNet-50: Layer 4 Removed")
        # print(self.base_model)

        # only training BatchNorm layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):        
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        print('x shape:', x.shape)

        # Extract features
        features = self.base_model(x)

        # Restore temporal dimension
        feature_channels, h, w = features.shape[1:]
        features = features.view(batch_size, num_frames, feature_channels, h, w)
        print('features shape:', features.shape)

        return features
    
class ModelWrapper(nn.Module):
    def __init__(self, do_not_reduce_frame_rate=False, dont_stack=False, small_embedder=False):
        super(ModelWrapper, self).__init__()

        self.cnn = BaseModel()
        self.emb = ConvEmbedder(dont_stack=dont_stack, small_embedder=small_embedder)
        self.do_not_reduce_frame_rate = do_not_reduce_frame_rate
        self.dont_stack = dont_stack

    def forward(self, data):
        
        # Pass through resnet50
        cnn_feats = self.cnn(data)

        # stack features
        context_frames = 3
        batch_size, num_frames, channels, feature_h, feature_w = cnn_feats.shape
        if self.dont_stack:
            num_context = num_frames
        else:
            if self.do_not_reduce_frame_rate:
                cnn_feats_temp = torch.zeros(batch_size, num_frames, context_frames, channels, feature_h, feature_w)
                pad = context_frames // 2
                padded_cnn_feats = F.pad(cnn_feats, (0, 0, 0, 0, 0, 0, pad, pad))
                cnn_feats_temp = torch.cat([padded_cnn_feats[:, i:i + num_frames] for i in range(context_frames)], dim=2)
                cnn_feats_temp = cnn_feats_temp.permute(0, 2, 1, 3, 4, 5)
                cnn_feats = cnn_feats_temp.view(batch_size * num_frames, context_frames, channels, feature_h, feature_w)
            else:
                num_context = num_frames // context_frames
                cnn_feats = cnn_feats[:, :num_context*context_frames, :, :, :]
                cnn_feats = cnn_feats.reshape(batch_size * num_context, context_frames, channels, feature_h, feature_w)

        # Pass CNN features through Embedder
        embs = self.emb(cnn_feats)          # cnn_feats: (batch_size, num_frames, channels, feature_h, feature_w)

        # Reshape to (batch_size, num_frames, embedding_dim)
        channels = embs.shape[-1]
        embs = embs.view(batch_size, num_context, channels)
        
        return embs
    
    def load_state_dict(self, state_dict):
        # Convert keys from 'cnn.' prefix to 'cnn' dict
        # Create new state dict without 'model.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        # Call parent class implementation with cleaned state dict
        super().load_state_dict(new_state_dict)

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
    model = ModelWrapper()
    output = model(dummy_data)
    print("Output shape:", output.shape)  # Expected: [batch_size, num_frames//3, embedding_dim]

if __name__ == "__main__":
    # test_baseModel()
    # test_convEmbedder()
    test_algorithm()
