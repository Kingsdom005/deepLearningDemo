import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class Pokemon(Dataset):
    def __init__(self, root, resize=None, mode='train'):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        
        print(self.name2label)
        
        self.images, self.labels = self.load_csv("images.csv")
        
        # use 60% data for training
        if mode == 'train':
            self.images = self.images[:int(len(self.images)*0.6)]
            self.labels = self.labels[:int(len(self.labels)*0.6)]
        # use 20% data for validation
        elif mode == 'val':
            self.images = self.images[int(len(self.images)*0.6):int(len(self.images)*0.8)]
            self.labels = self.labels[int(len(self.labels)*0.6):int(len(self.labels)*0.8)]
        # use 20% data for testing
        else:
            self.images = self.images[int(len(self.images)*0.8):]
            self.labels = self.labels[int(len(self.labels)*0.8):]
        
    def load_csv(self, filename):
        
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, "*.png"))
                images += glob.glob(os.path.join(self.root, name, "*.jpg"))
                images += glob.glob(os.path.join(self.root, name, "*.jpeg"))
            # 1167, paths
            # print(len(images), images)
            
            random.shuffle(images)
            with open(os.path.join(self.root, filename), 'w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])        
            print("csv saved to", os.path.join(self.root, filename))
        else:
            print("{} detected".format(os.path.join(self.root, filename)))
        
        print("csv loaded from", os.path.join(self.root, filename))
        images, labels = [], []
        with open(os.path.join(self.root, filename), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels
                
    
    def __len__(self):
        return len(self.images)
    
    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
        # x_hat = (x - mean) / std
        # x = x_hat * std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x
    
    
    def __getitem__(self, idx):
        # idx: [0, len(images)]
        # self.images, self.labels
        # 'pokemon\\bulbasaur\\00000000.png', 0
        img ,label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'), # string path to PIL image
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label
    
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import visdom, time
    
    viz = visdom.Visdom()
    
    """等价"""
    import torchvision
    
    db = torchvision.datasets.ImageFolder(root="pokemon", transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    
    loader = DataLoader(db, batch_size=32, shuffle=True)
    
    print(db.class_to_idx)
    
    for x, y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='batch_y', opts=dict(title='batch_y'))
        time.sleep(2)
    
    """等价"""
    
    # # 确保当前工作目录是脚本所在目录
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # db = Pokemon(root="pokemon",resize=224 , mode='train')
    
    # x, y  = next(iter(db))
    # print("sample:", x.shape, y.shape, y)
    
    # viz.image(db.denormalize(x), win='sample', opts=dict(title='sample_x'))
    
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    
    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='batch_y', opts=dict(title='batch_y'))
    #     time.sleep(10)
    
    
# if __name__ == '__main__':
#     main()