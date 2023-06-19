import torch
import torch.optim as optim
import torch.nn as nn
import config
from model import yolov3_backbone,segment,object_det
from dataloader import CarvanaDataset,YOLODataset
from loss import YoloLoss

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = "data/train_images/"
# TRAIN_MASK_DIR = "data/train_masks/"
# VAL_IMG_DIR = "data/val_images/"
# VAL_MASK_DIR = "data/val_masks/"

back_bone = yolov3_backbone()
model_seg = segment()
model_obj = object_det(num_classes=20)



segment_train = CarvanaDataset()
object_det_train = YOLODataset()


for epoch in range(NUM_EPOCHS):
    # 交替训练物体检测和语义分割模型

    if epoch % 2 == 0:
        model_obj.train()
        model_seg.eval()
        optimizer = optim.Adam([
                {'params': back_bone.parameters()},
                {'params': model_obj.parameters()}
            ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        current_model = model_obj
        dataloader = YOLODataset
        current_loss = YoloLoss()
        task_name = "Object Detection"
    else:
        model_obj.eval()
        model_seg.train()
        optimizer = optim.Adam([
                {'params': back_bone.parameters()},
                {'params': model_seg.parameters()}
            ], lr=LEARNING_RATE)
        current_model = model_seg
        dataloader = CarvanaDataset
        current_loss = nn.BCEWithLogitsLoss()
        task_name = "Semantic Segmentation"

    for batch_data in dataloader:
        # 前向传播
        inputs, labels = batch_data

        outputs = current_model(inputs)

        # 计算损失
        loss = current_loss(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印训练信息
    print(f"Epoch {epoch + 1} completed. {task_name} Loss: {loss.item()}")

