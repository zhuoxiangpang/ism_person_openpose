
from torch import nn, randn, exp, sum

class NetV1(nn.Module):

    def __init__(self):
        super().__init__()

        self.W = nn.Parameter(randn(16384,2))

    # 前项过程逻辑
    def forward(self, x):
       h = x@self.W
       # soft max
       h = exp(h)
       z = sum(h,dim=1,keepdim=True) #保持梯度
       return h/z

class NetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(16384,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):

        return self.sequential(x)

