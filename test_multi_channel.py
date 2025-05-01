import os
import torch
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn as nn
from collections import defaultdict
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

def load_model(model_path, model_class, device, **model_kwargs):
    model = model_class(**model_kwargs).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint with Train Loss: {checkpoint['train_loss']}")
    print(f"Loaded checkpoint with Train Accuracy: {checkpoint['train_accuracy']}")
    print(f"Loaded checkpoint with Val Loss: {checkpoint['val_loss']}")
    print(f"Loaded checkpoint with Val Accuracy: {checkpoint['val_accuracy']}")
    model.eval()
    return model

def preprocess_and_combine(mask_path, csv_data, original_dir, transform):
    # Extract filename without the "reconstructed_" prefix
    mask_name = os.path.basename(mask_path)
    csv_lookup_name = mask_name.replace("reconstructed_", "")  # Remove the prefix

    # Debug: Print the lookup name
    # print(f"Looking for: {csv_lookup_name} in CSV")

    # Find the corresponding row in the CSV
    row = csv_data[csv_data['picture'] == csv_lookup_name]
    
    if row.empty:
        raise ValueError(f"No corresponding entry found in CSV for {csv_lookup_name}")
    
    # Get the location ID and original image name
    location_id = int(row.iloc[0]['locationID']) - 1  # Adjust for zero-indexed IDs
    original_name = row.iloc[0]['picture']
    
    # Load original and masked images
    original_path = os.path.join(original_dir, original_name)
    original_image = Image.open(original_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('RGB')
    
    # Apply transformations
    if transform:
        original_image = transform(original_image)
        mask_image = transform(mask_image)
    
    # Combine the images along the channel dimension
    combined_image = torch.cat((original_image, mask_image), dim=0)
    
    return combined_image, location_id

def preprocess_and_combine_new(original_path, mask_path, transform):
    """
    Loads the original and mask images, applies the transform, and combines them along the channel dimension.
    
    Parameters:
      - original_path: Path to the original image.
      - mask_path: Path to the corresponding mask image.
      - transform: A torchvision transform to be applied to both images.
    
    Returns:
      - combined_image: A torch tensor combining both images along the channel dimension.
    """
    from PIL import Image
    import torch

    # Load original and masked images
    original_image = Image.open(original_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('RGB')
    
    # Apply transformations if provided
    if transform:
        original_image = transform(original_image)
        mask_image = transform(mask_image)
    
    # Combine the images along the channel dimension
    combined_image = torch.cat((original_image, mask_image), dim=0)
    
    return combined_image

def predict_with_location(model, combined_image, location_id, class_labels, device, criterion=None, target=None):
    """
    Predicts the class label using the model and optionally computes the loss.
    
    Parameters:
      - model: the trained model.
      - combined_image: the input tensor (without batch dimension).
      - location_id: camera ID (should be adjusted to zero-indexed if needed).
      - class_labels: list or mapping of class labels.
      - device: torch device.
      - criterion (optional): loss function. If provided along with a target, loss is computed.
      - target (optional): ground truth label (as an integer) for loss computation.
      
    Returns:
      - predicted_label: the predicted label (from class_labels).
      - loss_val (optional): the computed loss if criterion and target are provided, otherwise None.
    """
    # Add batch dimension and move to device
    combined_image = combined_image.unsqueeze(0).to(device)
    location_id_tensor = torch.tensor([location_id], dtype=torch.long).to(device)
    
    # Forward pass
    outputs = model(combined_image, location_id_tensor)
    _, predicted = outputs.max(1)  # Get the predicted class index
    predicted_label = class_labels[predicted.item()]
    
    loss_val = None
    # If a criterion and target are provided, compute loss.
    if criterion is not None and target is not None:
        target_tensor = torch.tensor([target], dtype=torch.long).to(device)
        loss_val = criterion(outputs, target_tensor)
    
    return predicted_label, loss_val

def test_dataset_with_combined_inputs(model, mask_dir, csv_file, original_dir, transform, class_labels, device):
    from collections import defaultdict

    csv_data = pd.read_csv(csv_file)
    total_images = 0
    correct_predictions = 0

    per_class_counts = defaultdict(int)   # Total samples per class
    per_class_correct = defaultdict(int)  # Correct predictions per class

    for label_folder in os.listdir(mask_dir):
        label_path = os.path.join(mask_dir, label_folder)

        if not os.path.isdir(label_path):
            continue

        for mask_name in os.listdir(label_path):
            mask_path = os.path.join(label_path, mask_name)

            try:
                combined_image, location_id = preprocess_and_combine(mask_path, csv_data, original_dir, transform)
            except ValueError as e:
                print(e)
                continue

            prediction = predict_with_location(model, combined_image, location_id, class_labels, device)

            per_class_counts[int(label_folder)] += 1  # Ensure label is treated as integer
            if prediction == label_folder:
                per_class_correct[int(label_folder)] += 1

            total_images += 1
            if prediction == label_folder:
                correct_predictions += 1

    # Calculate overall accuracy
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images} correct)")
    else:
        print("\nNo valid images found for testing. Please check the input paths and CSV mappings.")

    # Store accuracy per class in a list
    accuracy_list = []
    for label, total in per_class_counts.items():
        correct = per_class_correct[label]
        class_accuracy = (correct / total) * 100 if total > 0 else 0
        accuracy_list.append((int(label), class_accuracy, correct, total))  # Convert label to int

    # **Sort by accuracy (ascending order)**
    accuracy_list.sort(key=lambda x: x[0])  # Sort based on accuracy

    # Print sorted per-class accuracy
    print("\nPer-Class Accuracy (Sorted by Accuracy):")
    for label, class_accuracy, correct, total in accuracy_list:
        print(f"Class {label}: {class_accuracy:.2f}% ({correct}/{total})")

def test_dataset_with_combined_inputs_new(model, masks_dir, csv_file, images_dir, transform, class_labels, device):
    """
    Evaluate the model on test data using the new data organization.
    
    Parameters:
      - model: the trained model.
      - masks_dir: directory for masked images (e.g., /.../test_classification/masks).
      - csv_file: path to CSV with columns: locationID, picture, channelHeight(feet above sea level)
      - images_dir: directory for original images (e.g., /.../test_classification/images).
      - transform: preprocessing transformation (applied to both original and masked images).
      - class_labels: list or mapping of classes expected by your model.
      - device: torch device.
    """
    # Load CSV data
    csv_data = pd.read_csv(csv_file)
    
    total_images = 0
    correct_predictions = 0
    total_loss = 0.0
    per_class_counts = defaultdict(int)   # Total samples per class
    per_class_correct = defaultdict(int)  # Correct predictions per class

    predictions_list = []

    # Iterate over each row in the CSV file
    for idx, row in csv_data.iterrows():
        # Get the label, locationID, and picture name from the CSV.
        gt_label = int(row['channelHeight(feet above sea level)'])
        label = str(gt_label)  # label as an integer
        location_id = row['locationID'] - 1
        picture = row['picture']

        # Build the file paths:
        # For the original image:
        original_path = os.path.join(images_dir, picture)
        # For the mask image, assuming they follow the "reconstructed_{picture}" naming:
        mask_path = os.path.join(masks_dir, f"reconstructed_{picture}")

        try:
            # Use a helper function to load the original image and mask, apply transform, and combine them.
            # This function should return a combined tensor (e.g., 6 channels) ready for prediction.
            combined_image = preprocess_and_combine_new(original_path, mask_path, transform)
        except Exception as e:
            print(f"Error processing {picture}: {e}")
            continue
        
        criterion = nn.CrossEntropyLoss()
        # Make prediction using the model and the location_id (if your model uses it).
        prediction, loss_value = predict_with_location(model, combined_image, location_id, class_labels, device, criterion, gt_label)
        total_loss += loss_value.item()
        # Update per-class counts and overall counts.
        per_class_counts[label] += 1
        if prediction == label:
            per_class_correct[label] += 1
        total_images += 1
        if prediction == label:
            correct_predictions += 1
        predictions_list.append((picture, label, prediction))
    # Calculate overall accuracy.
    if total_images > 0:
        overall_loss = total_loss / total_images
        overall_accuracy = (correct_predictions / total_images) * 100
        print(f"\nOverall Test Loss: {overall_loss:.8f}, Overall Test Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_images} correct)")
    else:
        print("\nNo valid images found for testing. Please check the input paths and CSV mappings.")

    # Calculate per-class accuracy and store in a list.
    accuracy_list = []
    for cls, total in per_class_counts.items():
        correct = per_class_correct[cls]
        class_accuracy = (correct / total) * 100 if total > 0 else 0
        accuracy_list.append((cls, class_accuracy, correct, total))

    # Sort by class (or by accuracy, if you prefer)
    accuracy_list.sort(key=lambda x: x[0])  # sorting by class label

    # Print sorted per-class accuracy.
    print("\nPer-Class Accuracy (Sorted by Class):")
    for cls, class_accuracy, correct, total in accuracy_list:
        print(f"Class {cls}: {class_accuracy:.2f}% ({correct}/{total})")

    # print("\nSample Predictions:")
    # for sample in predictions_list[:20]:  # Print first 20 samples (or choose random ones)
    #     pic, ground_truth, pred = sample
    #     print(f"Image: {pic} | Ground Truth: {ground_truth} | Predicted: {pred}")

    return overall_accuracy, accuracy_list

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
            in_chans=6
        )
        # We determined via a dummy pass that the actual feature dimension is 16.
        self.backbone.train()
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

# Paths
# model_path = "best_model_UNet.pth"  # Path to your saved model
mask_dir = "/home/billyqu/flood/UNet/unet_classification/dataset/test_classification/masks"
original_dir = "/home/billyqu/flood/UNet/unet_classification/dataset/test_classification/images"
csv_file = "/home/billyqu/flood/UNet/unet_classification/dataset/test_classification/test_labels.csv"

# # Transformations
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a model with combined (original+mask) inputs and camera ID."
    )
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "efnet", "swinv2", "coatnet", "eva", "mamba", "resnet", "vit", "rs", "resmlp"],
                        help="Model to test: 'unet' or 'enhanced'.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g., best_model_UNet.pth).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 26
    num_cameras = 7  # Update as needed

    # Define the image transformations (should match training)
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
    # Define class labels (in this example, labels "0" through "25")
    class_labels = [str(i) for i in range(num_classes)]

    # Choose the model based on the command-line argument
    if args.model == "unet":
        model = load_model(args.model_path, UNetClassifier, device=device,
                           n_channels=6, n_classes=num_classes, num_cameras=num_cameras)
    elif args.model == "efnet":
        model = load_model(args.model_path, EnhancedClassifier, device=device,
                           num_classes=num_classes, num_cameras=num_cameras)
    elif args.model == "swinv2":
        model = load_model(args.model_path, SwinTransformerClassifier, device=device,
                           num_classes=num_classes, num_cameras=num_cameras)
    elif args.model == "coatnet":
        model = load_model(args.model_path, CoAtNetClassifier, device=device,
                           num_classes=num_classes, num_cameras=num_cameras)
    elif args.model == "eva":
        model = load_model(args.model_path, EVAClassifier, device=device,
        num_classes=num_classes, num_cameras=num_cameras, variant='eva02_large_patch14_224')
    elif args.model == "mamba":
        model = load_model(args.model_path, MambaClassifier, device=device,
                        num_classes=num_classes, num_cameras=num_cameras, variant='mambaout_base')
    elif args.model == "resnet":
        model = load_model(args.model_path, ResNetClassifier, device=device,
                        num_classes=num_classes, num_cameras=num_cameras, resnet_variant='resnet50')
    elif args.model == "vit":
        model = load_model(args.model_path, ViTClassifier, device=device,
                        num_classes=num_classes, num_cameras=num_cameras, variant='vit_base_patch16_224')
    elif args.model == "rs":
        model = load_model(args.model_path, ResNetRSClassifier, device=device,
                        num_classes=num_classes, num_cameras=num_cameras, resnet_variant='resnetrs50')
    elif args.model == "resmlp":
        model = load_model(args.model_path, ResMLPClassifier, device=device,
                       num_classes=num_classes, num_cameras=num_cameras, variant='resmlp_12_224')
    else:
        raise ValueError("Invalid model selection!")

    print(f"Testing using the '{args.model}' model...")
    test_dataset_with_combined_inputs_new(model, mask_dir, csv_file,
                                      original_dir, transform, class_labels, device)