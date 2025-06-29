from model import FCN
import torch
from torchvision.datasets import VOCSegmentation
from dataset import SegmentationDataset
from util import visualize_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VOC sample dataset
voc = VOCSegmentation(root="segmentation/data", year="2012", image_set="train", download=True)

image_paths = [img for img, _ in voc]
mask_paths = [mask for _, mask in voc]

# Use a subset for quick testing
dataset = SegmentationDataset(image_paths[:100], mask_paths[:100])

num_classes = 21  # Pascal VOC has 21 classes
model = FCN(num_classes=num_classes)
model.load_state_dict(torch.load("segmentation/model.pth"))

model.eval()
img, mask = dataset[2]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()
visualize_prediction(img, pred_mask)
