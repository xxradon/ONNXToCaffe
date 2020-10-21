import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.relu = nn.ReLU6()

    def forward(self,x):
        return self.relu(x)

torch.onnx.export(model(),torch.zeros([1,3,224,224]),"test.onnxi")
