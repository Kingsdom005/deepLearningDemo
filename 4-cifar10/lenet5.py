import torch
import torch.nn as nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self):
        super(Lenet5, self).__init__()
        
        self.conv_unit = nn.Sequential(
            # [b, 3, 32, 32] => [b, 6, 28, 28]
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            # [b, 6, 28, 28] => [b, 6, 14, 14]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, 14, 14] => [b, 16, 10, 10]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            # [b, 16, 10, 10] => [b, 16, 5, 5]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        
        self.fc_unit = nn.Sequential(
            # [b, 16, 5, 5] => [b, 120]
            # nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),#in_features=16*5*5 not 2
            nn.ReLU(),
            # [b, 120] => [b, 84]
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # [b, 84] => [b, 10]
            nn.Linear(in_features=84, out_features=10)
        )
        
        # test
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print("conv out shape: ", out.shape)
        
        # use CrossEntropyLoss(include softmax)
        # self.criteon = nn.CrossEntropyLoss()
        
        
    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return: [b, 10]
        """
        batchsz = x.size(0)
        #[b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # flatten [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 16*5*5)
        #[b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        
        # # [b, *10]
        # pred = F.softmax(logits, dim=1)
        
        # outer layer calc loss
        # loss = self.criteon(logits, y)
        return logits
            
        
def main():
    net = Lenet5()
    # test
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print("conv out shape: ", out.shape)
        
if __name__ == '__main__':
    main()