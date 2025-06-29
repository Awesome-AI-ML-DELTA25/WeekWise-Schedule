import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from dataset import SegmentationDataset
from model import FCN
from train import train_model
from util import visualize_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VOC sample dataset
voc = VOCSegmentation(root="segmentation/data", year="2012", image_set="train", download=True)

image_paths = [img for img, _ in voc]
mask_paths = [mask for _, mask in voc]

# Use a subset for quick testing
dataset = SegmentationDataset(image_paths[:100], mask_paths[:100])
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss, optimizer
num_classes = 21  # Pascal VOC has 21 classes
model = FCN(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
train_model(model, dataloader, criterion, optimizer, device, num_classes)
torch.save(model.state_dict(), "segmentation/model.pth")
# Visualize one prediction
model.eval()
img, mask = dataset[0]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()
visualize_prediction(img, pred_mask)
