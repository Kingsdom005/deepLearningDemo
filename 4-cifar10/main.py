import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from lenet5 import Lenet5

def main():
    batch_size = 128
    
    cifar_train = datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=transforms.Compose([
       transforms.Resize((32, 32)),
       transforms.ToTensor() 
    ]))
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    
    cifar_test = datasets.CIFAR10(root='./cifar10_data', train=False, download=False, transform=transforms.Compose([
       transforms.Resize((32, 32)),
       transforms.ToTensor() 
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    
    x, label = next(iter(cifar_train))
    print("x.shape: ",x.shape, "label: ", label.shape)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)
    
    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # x : [batch_size, 3, 32, 32]
            # label : [batch_size]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("train:", epoch, loss.item())            
        
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()  #(pred == label).sum().item()
                total_num += x.shape[0]
            acc = total_correct / total_num
            print("test:", epoch, acc)
            
    
if __name__ == '__main__':
    main()
    