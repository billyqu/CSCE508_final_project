import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from tqdm import tqdm
import argparse
import timm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # 使用双线性插值上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        # 调整上采样后的 x1 和下采样时保存的 x2 尺寸一致
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # 拼接和卷积
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        # 下采样：逐步缩小特征图尺寸，提取高层次特征
        self.inc = DoubleConv(n_channels, 64)  # 输入 -> 64 通道
        self.down1 = Down(64, 128)  # 64 -> 128 通道
        self.down2 = Down(128, 256)  # 128 -> 256 通道
        self.down3 = Down(256, 512)  # 256 -> 512 通道
        factor = 2 if bilinear else 1  # 双线性插值时调整特征图大小
        self.down4 = Down(512, 1024 // factor)  # 512 -> 1024 或 512
 
        # 上采样：逐步恢复原始尺寸，保留上下文信息
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
 
        # 最后一层输出卷积
        self.outc = OutConv(64, n_classes)
 
    def forward(self, x):
        # 前向传播路径
        x1 = self.inc(x)  # 初始卷积
        x2 = self.down1(x1)  # 第一次下采样
        x3 = self.down2(x2)  # 第二次下采样
        x4 = self.down3(x3)  # 第三次下采样
        x5 = self.down4(x4)  # 第四次下采样
        x = self.up1(x5, x4)  # 第一次上采样
        x = self.up2(x, x3)  # 第二次上采样
        x = self.up3(x, x2)  # 第三次上采样
        x = self.up4(x, x1)  # 第四次上采样
        logits = self.outc(x)  # 输出卷积层
        return logits  # 返回分割结果


# Define the Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(f"Accessing idx: {idx}, Dataset size: {len(self.df)}")
        row = self.df.iloc[idx]
        camera_id = row['locationID'] - 1
        image_name = row['picture']
        label = row['channelHeight(feet above sea level)']

        # Load original image (RGB)
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # Convert to RGB (3 channels)

        # Load masked image (RGB)
        mask_name = f"reconstructed_{image_name}"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('RGB')  # Convert to RGB (3 channels)

        # Apply transformations
        if self.transform:
            # rgb_transform = transforms.Compose([
            #     transforms.Resize((512, 512)),  # Resize the images
            #     transforms.ToTensor(),  # Convert to tensor
            #     transforms.Normalize([0.485, 0.456, 0.406],  # Mean for RGB
            #                          [0.229, 0.224, 0.225])  # Std for RGB
            # ])
            # image = rgb_transform(image)
            # mask = rgb_transform(mask)
            image = self.transform(image)
            mask = self.transform(mask)

        # Combine original image and mask into a single tensor
        combined_image = torch.cat((image, mask), dim=0)  # Combine along channel dimension

        # Validate combined image shape
        assert combined_image.shape[0] == 6, f"Expected 6 channels, got {combined_image.shape[0]}"

        return combined_image, camera_id, label

# Transformations
# print("def transform to 224")
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225])
# ])

parser = argparse.ArgumentParser(
    description="Train a model with combined (original+mask) images and camera ID."
)
parser.add_argument("--model", type=str, default="unet",
                choices=["unet", "efnet", "swinv2", "coatnet", "eva", "mamba", "resnet", "vit", "rs", "resmlp"],
                )
args = parser.parse_args()
if args.model in ["vit", "resmlp", "unet", "efnet"]:
    print("def transform to 224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Use 224x224 for mambaout_base
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
else:
    print("def transform to 512")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

# Initialize Dataset
train_data_path = "/home/billyqu/flood/UNet/unet_classification/dataset/train_classification/images"
mask_data_path = "/home/billyqu/flood/UNet/unet_classification/dataset/train_classification/masks"
csv_file = "/home/billyqu/flood/UNet/unet_classification/dataset/train_classification/train_labels.csv"
dataset = CustomDataset(csv_file=csv_file, image_dir=train_data_path, mask_dir=mask_data_path, transform=transform)

# Split the Dataset
indices = list(range(len(dataset)))
labels = [label for _, _, label in dataset]

from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# Step 1: Count the samples for each class
class_counts = Counter(labels)
print("Class counts:", class_counts)

# Step 2: Identify classes with less than 2 samples
single_sample_classes = [label for label, count in class_counts.items() if count < 2]
print("Classes with less than 2 samples:", single_sample_classes)

# Step 3: Pad single-sample classes dynamically
expanded_indices = []
expanded_labels = []

for idx, label in zip(indices, labels):
    expanded_indices.append(idx)
    expanded_labels.append(label)
    if label in single_sample_classes:
        # Duplicate the sample to create at least 2 samples per class
        expanded_indices.append(idx)
        expanded_labels.append(label)

expanded_dataset = torch.utils.data.Subset(dataset, expanded_indices)

# Step 4: Perform stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_indices = next(splitter.split(expanded_indices, expanded_labels))

print(f"Total dataset size: {len(dataset)}")
print(f"Train dataset size: {len(train_indices)}")
print(f"Validation dataset size: {len(val_indices)}")

# Debugging: Print distributions after padding and splitting
train_labels = [expanded_labels[i] for i in train_indices]
val_labels = [expanded_labels[i] for i in val_indices]

print("Training class distribution:", Counter(train_labels))
print("Validation class distribution:", Counter(val_labels))


train_dataset = torch.utils.data.Subset(expanded_dataset, train_indices)
val_dataset = torch.utils.data.Subset(expanded_dataset, val_indices)

print(f"Original dataset size: {len(dataset)}")
print(f"Expanded dataset size: {len(expanded_indices)}")
print(f"Train dataset size: {len(train_indices)}")
print(f"Validation dataset size: {len(val_indices)}")

# DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the Model
class EnhancedClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16):
        super(EnhancedClassifier, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 6 channels
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=6,  # Updated to accept 6 channels (RGB + RGB)
            out_channels=self.backbone.features[0][0].out_channels,
            kernel_size=self.backbone.features[0][0].kernel_size,
            stride=self.backbone.features[0][0].stride,
            padding=self.backbone.features[0][0].padding,
            bias=False
        )

        # Camera ID embedding
        self.camera_id_embedding = nn.Embedding(num_embeddings=7, embedding_dim=embed_dim)

        # Classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove the default classifier
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, camera_id):
        # Pass the image through the backbone
        features = self.backbone(x)

        # Embed the camera ID
        camera_features = self.camera_id_embedding(camera_id)

        # Concatenate features
        combined_features = torch.cat((features, camera_features), dim=1)

        # Pass through the classification head
        return self.fc(combined_features)

class UNetClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, num_cameras, embed_dim=16):
        super(UNetClassifier, self).__init__()

        # Image Feature Extraction (U-Net Encoder)
        self.encoder = nn.Sequential(
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024)
        )

        # Global Average Pooling to Reduce Dimensionality
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  

        # Camera ID Embedding Layer
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)

        # Attention Mechanism: Determines how much weight to assign to camera ID
        self.attention_fc = nn.Linear(embed_dim, 1024)  # Maps camera embedding to feature space

        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),  # Uses adjusted image features after fusion
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, camera_id):
        # Extract Image Features
        x = self.encoder(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten before FC
        # Embed Camera ID
        camera_features = self.camera_embedding(camera_id)  # Shape: (batch_size, embed_dim)

        # Transform camera_features to match x's feature size
        camera_features = self.attention_fc(camera_features)  # Now shape is (batch_size, 1024)
    
        # Learnable Attention: Map camera embedding to image feature size
        attention_weights = torch.sigmoid(camera_features)  # Outputs values in [0,1]

        # Weighted Fusion (Dynamically Adjusts Contribution)
        fused_features = x * attention_weights + camera_features  # Element-wise fusion

        # Classification
        return self.fc(fused_features)

class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16):
        super(SwinTransformerClassifier, self).__init__()
        # Create a Swin Transformer V2 model with 6 input channels
        # You can choose a different variant if desired
        self.backbone = timm.create_model(
            'swinv2_base_window8_256',  # example variant; adjust as needed
            pretrained=True,
            in_chans=6,
            img_size=512
        )
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()  # Remove the default classifier head
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, camera_id):
        features = self.backbone.forward_features(x)
        # print("features.shape:", features.shape)
        if features.dim() == 4:
            features = features.mean(dim=(1,2))
        camera_features = self.camera_embedding(camera_id)
        combined_features = torch.cat((features, camera_features), dim=1)
        return self.fc(combined_features)

class CoAtNetClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16):
        super(CoAtNetClassifier, self).__init__()
        # Create a CoAtNet model with 6 input channels
        # variant can be 'coatnet_0', 'coatnet_1', etc.
        self.backbone = timm.create_model(
            "coatnet_2_rw_224",
            pretrained=True,
            in_chans=6,
            img_size=512
        )
        # Remove the default classification head and obtain feature dimension
        # Many timm models support reset_classifier; we do that here.
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x, camera_id):
        # Use forward_features to get a feature vector
        features = self.backbone.forward_features(x)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))  # Now shape becomes [B, C]
        camera_features = self.camera_embedding(camera_id)
        combined_features = torch.cat((features, camera_features), dim=1)
        return self.fc(combined_features)

class EVAClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, variant='eva02_large_patch14_224'):
        super(EVAClassifier, self).__init__()
        self.backbone = timm.create_model(
            variant,
            pretrained=True,
            in_chans=6,
            img_size=512
        )
        # Instead of using self.backbone.num_features, use the confirmed value:
        in_features = 1297  # Hard-coded after verifying with a dummy pass
        self.backbone.reset_classifier(0)
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, camera_id):
        features = self.backbone.forward_features(x)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            features = features.mean(dim=2)
        camera_features = self.camera_embedding(camera_id)
        combined_features = torch.cat((features, camera_features), dim=1)
        return self.fc(combined_features)

class MambaClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, variant='mambaout_base'):
        super(MambaClassifier, self).__init__()
        # Create the mamba backbone.
        self.backbone = timm.create_model(
            variant,
            pretrained=True,
            in_chans=6  # mambaout_base does not accept an img_size parameter.
        )
        # We determined via a dummy pass that the actual feature dimension is 16.
        in_features = 16  
        # Remove the default classification head.
        self.backbone.reset_classifier(0)
        # Define camera ID embedding.
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        # Build the FC head: input dimension = in_features + embed_dim.
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),  # Here: 16 + embed_dim.
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, camera_id):
        features = self.backbone.forward_features(x)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            features = features.mean(dim=2)
        camera_features = self.camera_embedding(camera_id)
        combined_features = torch.cat((features, camera_features), dim=1)
        return self.fc(combined_features)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, resnet_variant='resnet50'):
        super(ResNetClassifier, self).__init__()
        # Load a pretrained ResNet model.
        resnet = getattr(models, resnet_variant)(pretrained=True)
        
        # Modify the first convolution layer to accept 6 channels.
        # Original conv1: in_channels=3, out_channels=64, etc.
        resnet.conv1 = nn.Conv2d(
            6,
            resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False
        )
        
        # Get the feature dimension from the original fc layer.
        in_features = resnet.fc.in_features
        
        # Remove the default classifier head.
        resnet.fc = nn.Identity()
        
        self.backbone = resnet
        # Define camera ID embedding.
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        # Define a new classification head that fuses the ResNet features and camera embedding.
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, camera_id):
        # Pass the image through ResNet; output shape is [B, in_features].
        features = self.backbone(x)
        # Get the camera embedding; shape: [B, embed_dim].
        camera_features = self.camera_embedding(camera_id)
        # Concatenate the features and camera embedding.
        combined_features = torch.cat((features, camera_features), dim=1)
        return self.fc(combined_features)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, variant='vit_base_patch16_224'):
        super(ViTClassifier, self).__init__()
        # Create a ViT model with 6 input channels.
        self.backbone = timm.create_model(
            variant,
            pretrained=True,
            in_chans=6,    # Change to 6 channels for your combined input.
            img_size=224   # Use 224x224, as expected by the variant.
        )
        # Retrieve the feature dimension from the original classification head.
        in_features = self.backbone.head.in_features
        # Remove the default classification head.
        self.backbone.head = nn.Identity()
        
        # Define the camera ID embedding.
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        # Build the custom classifier head.
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, camera_id):
        # Pass the input through the ViT backbone. It should output a flattened feature vector.
        features = self.backbone(x)
        # Obtain the camera embedding.
        cam_emb = self.camera_embedding(camera_id)
        # Concatenate the ViT features and the camera embedding.
        combined_features = torch.cat((features, cam_emb), dim=1)
        return self.fc(combined_features)

class ResNetRSClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, resnet_variant='resnetrs50'):
        super(ResNetRSClassifier, self).__init__()
        resnet_rs = timm.create_model(resnet_variant, pretrained=True)

        new_conv = nn.Conv2d(
            in_channels=6,           # New input channels.
            out_channels=32,         # Force output to 32 channels (to match BatchNorm).
            kernel_size=resnet_rs.conv1[0].kernel_size,
            stride=resnet_rs.conv1[0].stride,
            padding=resnet_rs.conv1[0].padding,
            bias=False
        )
        resnet_rs.conv1[0] = new_conv
        in_features = resnet_rs.fc.in_features

        resnet_rs.fc = nn.Identity()
        
        self.backbone = resnet_rs

        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, camera_id):
        features = self.backbone(x)
        cam_emb = self.camera_embedding(camera_id)
        combined_features = torch.cat((features, cam_emb), dim=1)
        return self.fc(combined_features)

class ResMLPClassifier(nn.Module):
    def __init__(self, num_classes, num_cameras, embed_dim=16, variant='resmlp_12_224'):
        super(ResMLPClassifier, self).__init__()
        # Create a ResMLP model from timm.
        # Ensure that the model supports changing the number of input channels.
        self.backbone = timm.create_model(
            variant,
            pretrained=True,
            in_chans=6,      # Change from 3 to 6 channels.
            img_size=224     # ResMLP_12 is typically trained on 224x224 images.
        )
        # Retrieve the feature dimension.
        in_features = self.backbone.num_features  # This attribute should give you the flattened feature dimension.
        # Remove the default classification head.
        self.backbone.reset_classifier(0)
        
        # Define camera ID embedding.
        self.camera_embedding = nn.Embedding(num_embeddings=num_cameras, embedding_dim=embed_dim)
        
        # Define the custom classifier head that fuses ResMLP features with the camera embedding.
        self.fc = nn.Sequential(
            nn.Linear(in_features + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, camera_id):
        # Forward pass through the ResMLP backbone.
        features = self.backbone(x)  # Expect shape: [B, in_features]
        cam_emb = self.camera_embedding(camera_id)  # Shape: [B, embed_dim]
        combined_features = torch.cat((features, cam_emb), dim=1)  # Combined shape: [B, in_features + embed_dim]
        return self.fc(combined_features)

#train function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, best_model_path):
    best_train_loss = float('inf')
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=True)
        for images, camera_ids, labels in train_loader_tqdm:
            images, camera_ids, labels = images.to(device), camera_ids.to(device), labels.to(device)
            outputs = model(images, camera_ids)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        # if epoch_loss < best_train_loss:
        #     best_train_loss = epoch_loss
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': best_train_loss
        #     }, best_model_path)
        #     print(f"Best model saved with training loss: {best_train_loss:.4f}")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=True)
            for images, camera_ids, labels in val_loader_tqdm:
                images, camera_ids, labels = images.to(device), camera_ids.to(device), labels.to(device)
                outputs = model(images, camera_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation - Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        # if val_epoch_loss < best_val_loss:
        #     best_val_loss = val_epoch_loss
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': best_val_loss
        #     }, best_model_path)
        #     print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_epoch_loss,
                'val_accuracy': best_val_accuracy
            }, best_model_path)
            print(f"Best model saved with validation accuracy: {best_val_accuracy:.2f}%")


# Initialize Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a model with combined (original+mask) images and camera ID."
        )
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "efnet", "swinv2", "coatnet", "eva", "mamba", "resnet", "vit", "rs", "resmlp"],
                        )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    args = parser.parse_args()

    num_classes = 26
    num_cameras = len(dataset.df['locationID'].unique())+1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "unet":
        model = UNetClassifier(n_channels=6, n_classes=num_classes, num_cameras=num_cameras).to(device)
    elif args.model == "efnet":
        model = EnhancedClassifier(num_classes=num_classes, num_cameras=num_cameras).to(device)
    elif args.model == "swinv2":
        model = SwinTransformerClassifier(num_classes=num_classes, num_cameras=num_cameras).to(device)
    elif args.model == "coatnet":
        model = CoAtNetClassifier(num_classes=num_classes, num_cameras=num_cameras).to(device)
    elif args.model == "eva":
        model = EVAClassifier(num_classes=num_classes, num_cameras=num_cameras, variant='eva02_large_patch14_224').to(device)
    elif args.model == "mamba":
        model = MambaClassifier(num_classes=num_classes, num_cameras=num_cameras, variant='mambaout_base').to(device)
    elif args.model == "resnet":
        model = ResNetClassifier(num_classes=num_classes, num_cameras=num_cameras, resnet_variant='resnet50').to(device)
    elif args.model == "vit":
        model = ViTClassifier(num_classes=num_classes, num_cameras=num_cameras, variant='vit_base_patch16_224').to(device)
    elif args.model == "rs":
        model = ResNetRSClassifier(num_classes=num_classes, num_cameras=num_cameras, resnet_variant='resnetrs50').to(device)
    elif args.model == "resmlp":
        model = ResMLPClassifier(num_classes=num_classes, num_cameras=num_cameras, variant='resmlp_12_224').to(device)
    else:
        raise ValueError("Invalid model selection!")

    # dummy_batch, _, _ = next(iter(train_loader))
    # dummy_batch = dummy_batch.to(device)
    # with torch.no_grad():
    #     dummy_features = model.backbone.forward_features(dummy_batch)
    #     if dummy_features.dim() == 4:
    #         dummy_features = dummy_features.mean(dim=(2, 3))
    #     elif dummy_features.dim() == 3:
    #         dummy_features = dummy_features.mean(dim=2)
    # print("Runtime dummy features shape:", dummy_features.shape)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_model_path = "best_model.pth"  # Path to save the best model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, best_model_path)