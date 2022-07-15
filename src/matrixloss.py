import torch
class MatrixLoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MatrixLoss, self).__init__()

    def forward(self, output, target):
        loss_matrix = torch.sub(input=target, other=output, alpha=1)
        loss_matrix = torch.abs(loss_matrix)
        loss =torch.mean(loss_matrix)
        return loss