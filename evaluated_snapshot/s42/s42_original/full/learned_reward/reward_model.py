from torch import nn

class RewardHead(nn.Module):
    def __init__(self,input_dim=2068,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)
