import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        
        if ch_in != ch_out or stride != 1:  # 修改：增加 stride != 1 的条件
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(ch_out)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        # shortcut
        # element-wise addition
        extra_out = self.extra(x)  # 修改：将 self.extra(x) 的结果保存到变量中
        # print("extra_out shape:", extra_out.shape)  # 调试：打印 extra_out 的形状
        # print("out shape:", out.shape)  # 调试：打印 out 的形状
        out = extra_out + out  # 修改：确保形状一致
        
        return out
    

class ResNet18(nn.Module):
    
    def __init__(self):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        
        # [b,64,h,w] => [b,128,h,w]
        self.blk1 = ResBlk(64, 128, 2)
        
        # [b,128,h,w] => [b,256,h,w]
        self.blk2 = ResBlk(128, 256, 2)
        
        # [b,256,h,w] => [b,512,h,w]
        self.blk3 = ResBlk(256, 512, 2)
        
        # [b,512,h,w] => [b,512,h,w]
        self.blk4 = ResBlk(512, 512, 2)
        
        self.outlayer = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        
        # print("after conv:", x.shape)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print("after pool:", x.shape)
        x = x.view(x.size(0), -1)
        
        x = self.outlayer(x)
        return x
    
# def main():
#     blk = ResBlk(64, 128, 2)
#     tmp = torch.randn(2, 64, 32, 32)
#     out = blk(tmp)
#     print("block:", out.shape)
    
#     x = torch.randn(2, 3, 32, 32)
#     model = ResNet18()
#     out = model(x)
#     print("resnet18:", out.shape)
    
# if __name__ == '__main__':
#     main()