import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.relu = nn.ReLU6()

    def forward(self,x):
        return self.relu(x)

class model_upsample(nn.Module):
    def __init__(self):
        super(model_upsample,self).__init__()
        # self.relu = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.relu = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,x):
        return self.relu(x)


# torch.onnx.export(model(),torch.zeros([1,3,224,224]),"test.onnxi")
input = torch.zeros([1,3,12,12])
torch.onnx.export(model_upsample(),input,"test.onnx")#,opset_version=9
