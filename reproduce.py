import torch


inputs_0 = torch.load('inputs.pth')
print(inputs_0.is_contiguous())
inputs_1 = inputs_0.contiguous()
print(inputs_1.is_contiguous())


inputs_2 = inputs_0 / torch.std(inputs_0, 1)[:, None]
inputs_0 /= torch.std(inputs_0, 1)[:, None]
inputs_1 /= torch.std(inputs_1, 1)[:, None]


print(inputs_0[-1][0])
print(inputs_1[-1][0])
print(inputs_2[-1][0])
