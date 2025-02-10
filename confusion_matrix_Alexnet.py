import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader

# Khởi tạo mô hình AlexNet với 11 lớp đầu ra
model = models.alexnet(num_classes=11)

# Load trọng số từ checkpoint
checkpoint = torch.load('/kaggle/input/alexnet_after_train/pytorch/default/1/model_glass_best.pth', map_location='cpu')

# Kiểm tra checkpoint có state_dict không
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # Nếu checkpoint lưu trực tiếp state_dict

# Loại bỏ tiền tố "module." nếu có
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load vào model
model.load_state_dict(new_state_dict, strict=False)

def main():
    # Đặt chế độ đánh giá
    model.eval()
    
    traindir, train_anno = '/kaggle/input/peta-dataset/PETA_dataset', ''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Định nghĩa dataset (Cần có ColorAttribute)
    val_dataset = ColorAttribute(
        traindir,
        train_anno, 
        'test',
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ]))

    # Tạo DataLoader
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Dự đoán và tính ma trận nhầm lẫn
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)

    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) 

    # Nếu ColorAttribute không có classes, tự tạo danh sách nhãn
    class_names = ['Black', 'Grey', 'Blue', 'White', 'Brown', 'Red', 'Purple', 'Green', 'Pink', 'Orange', 'Yellow']  # Thay bằng nhãn thực tế

    # Vẽ biểu đồ heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of AlexNet")
    
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

# Gọi hàm main để chạy
main()
