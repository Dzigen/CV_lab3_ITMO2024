from ultralytics import YOLO
import torch
import torchvision

BASE_DIR = '/home/ubuntu/ImgGen/lab3'
SAVE_DIR = f'{BASE_DIR}/save_dir'
EPOCHS = 100
BATCH_SIZE = 16

model = YOLO(f'{BASE_DIR}/yolov8n-seg.pt')
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

results = model.train(data=f'{BASE_DIR}/data/config.yaml', epochs=EPOCHS, 
                      batch=BATCH_SIZE, save_dir=SAVE_DIR)

results2 = model.eval()