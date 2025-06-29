import torch
from tqdm import tqdm
from util import compute_iou

def train_model(model, dataloader, criterion, optimizer, device, num_classes):
    model.to(device)
    model.train()
    for epoch in range(100):
        total_loss = 0
        total_iou = 0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes)
            total_iou += iou

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, mIoU = {avg_iou:.4f}")
