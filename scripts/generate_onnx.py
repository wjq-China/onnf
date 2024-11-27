# # from typing import *
# import numpy as np
# import torch


# max_xi = []
# softmax_xi = []
# l_xi = []


# # Numpy
# def softmax(x: np.ndarray) -> np.ndarray:
#     global max_xi, softmax_xi, l_xi

#     max_x = np.max(x)
#     max_xi.append(max_x)

#     exp_x = np.exp(x - max_x)
#     l_x = np.sum(exp_x)
#     l_xi.append(l_x)

#     softmax_x = exp_x / l_x
#     softmax_xi.append(softmax_x)


# # 使用pytorch 的 softmax验证
# def softmax_torch(x: torch.Tensor) -> torch.Tensor:
#     return torch.nn.functional.softmax(x, dim=0)


# input = np.random.randn(20)
# parallel = 4  # 4个元素为一个子向量

# # pass1 计算每个子向量的softmax
# for i in range(0, len(input), parallel):
#     softmax(input[i : i + parallel])

# #  pass2 计算分母
# res = []
# max_x = max(max_xi)
# l_x = 0
# for i in range(int(input.shape[0] / parallel)):
#     l_x += l_xi[i] * np.exp(max_xi[i] - max_x)

# # pass3 计算最终的结果
# for i in range(int(input.shape[0] / parallel)):
#     softmax_new_xi = (softmax_xi[i] * l_xi[i] * np.exp(max_xi[i] - max_x)) / l_x
#     res.extend(softmax_new_xi)
# print(res)

# input_torch = torch.tensor(input, dtype=torch.float32)
# print(softmax_torch(input_torch).numpy())


######################
import torch
import torch.nn as nn


class MultiplyAddModel(nn.Module):
    def __init__(
        self,
    ):
        super(MultiplyAddModel, self).__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, a, b, c):
        d = a * b
        e = d + c
        f = self.linear(e)
        return f


# 创建模型实例
model = MultiplyAddModel()
# 准备固定大小的示例输入
a = torch.tensor([[1.0, 2.0]])  # 输入 a，形状 (1, 2)
b = torch.tensor([[4.0, 5.0]])  # 输入 b，形状 (1, 2)
c = torch.tensor([[7.0, 8.0]])  # 输入 c，形状 (1, 2)

output_file = "../results/add.onnx"
# 导出为 ONNX 模型
torch.onnx.export(
    model,  # PyTorch 模型
    (a, b, c),  # 示例输入
    output_file,  # 输出 ONNX 文件名
    export_params=True,  # 导出模型参数
    opset_version=11,  # ONNX opset 版本
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=["a", "b", "c"],  # 输入名
    output_names=["output"],  # 输出名
    dynamic_axes={  # 可选，支持动态输入
        "a": {0: "batch_size"},
        "b": {0: "batch_size"},
        "c": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print("ONNX 模型已生成：add.onnx")
