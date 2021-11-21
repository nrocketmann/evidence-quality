import torch
from torch import nn

#backbone is the model that predicts the [C,D] in the paper, e.g. a BERT with a linear layer on top
class Siamese(nn.Module):
    def __init__(self, backbone):
        super(Siamese,self).__init__()
        self.backbone = backbone

    def forward(self, input1, input2, return_logits = True):
        #forward on both legs of the network
        output1, output2 = self.backbone(*input1), self.backbone(*input2)
        #shape B x N where N is 1 or 2 in case of dummy output

        #if there's a dummy output, we don't need it to do softmax here. If not, we're just squeezing by doing this
        output1 = output1[...,0]
        output2 = output2[...,0]

        together = torch.stack([output1, output2],dim=-1)
        if return_logits:
            return together #should be used with CrossEntropyLoss (not NLL, since we didn't apply log_softmax)
        else:
            return torch.softmax(together,-1)