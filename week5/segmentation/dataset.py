from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx].convert("RGB")
        mask = self.mask_paths[idx]
        
        img = self.transform(img)
        mask = T.Resize((128, 128))(mask)
        mask = T.PILToTensor()(mask).long().squeeze(0)
        
        return img, mask
