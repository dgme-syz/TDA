from torch.utils.data import Dataset


class CombDataset(Dataset):
    def __init__(self):
        self.labels = []
        self.images = []
        self.nameLabel = {}

    def add(self, images, target: str):
        self.nameLabel[target] = len(self.labels)
        self.labels.append(target)
        self.images.append(images)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            "image": self.images[index], 
            "label": self.nameLabel[self.labels[index]]
        }
