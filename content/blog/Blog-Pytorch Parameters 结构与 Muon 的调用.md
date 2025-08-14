+++
title = "Pytorch Parameters 结构与 Muon 的调用"
date = "2025-08-08"

[taxonomies]
tags = ["PyTorch", "Muon", "Machine Learning"]
+++

# 

近期想测试 [Muon](https://github.com/KellerJordan/Muon) 优化器的效果，看了下 Muon 官方 Github 的实现，发现无法像 Adam 或者 AdamW 那样直接简单地调用。还是要学习一下 `pytorch`​ 中 `model.parameters()`​ 的结构，最终搞明白如何使用 `Muon`​ 优化器。

## 背景

众所周知，我们一般创建 `Adam`​ 等优化器时只需要把模型内部的参数 `model.parameters()`​ 传给优化器就可以很简单地创建一个优化器，例如以下代码：

```python
import torch
optimizer = optim.AdamW(model.parameters(), lr=lr)
```

虽然大多数优化器都能以上面这样简单的方式进行创建，`Muon`​ 却不太一样。`Muon`​ 是针对二维以及更高维张量的优化器，而剩余的一维张量（例如偏置、Embedding层、输出层）则交给 Adam。按照其官方仓库的说明，要把 `AdamW`​ 替换为 `Muon`​ 得进行一定程度的修改：

```python
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

# 要把 AdamW 替换为 Muon，则需要使用以下代码：

from muon import MuonWithAuxAdam
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
```

可以看出上面的代码涉及到了一些 `Pytorch`​ 中 `Parameters`​ 的结构，对于没咋写过优化器的炼金师（我）来说还是有点儿不熟悉的，而且可能要根据项目需求进行进一步修改。所以要正确使用 `Muon`​ 优化器，还是得充分理解 `Pytorch`​ 中 `Parameters`​ 的结构。

## Model.Parameters() 的本质

**Parameter 对象**：在 `Pytorch`​ 中，一个神经网络模型（对应 `torch.nn.Module`​ 类）由多个层组成，每个层又包含一些可以学习的参数（主要是权重 `weights`​ 和偏置 `bias`​），这些参数被封装为 `torch.nn.Parameter`​ 对象。`torch.nn.Parameter`​ 对象本质上是特殊的 `Tensor`​，其与普通 `Tensor`​ 的区别在于其被注册为模型的参数，在调用 `loss.backward()`​ 时 `Pytorch`​ 会自动计算其对应的梯度。

**model.parameters()** ：`model.parameters()`​ 返回一个迭代器（Iterator），我们可以用它遍历模型中注册的 `torch.nn.Parameter`​ 对象。大多数情况下，我们只需要使用这个迭代器就可以创建一个优化器，例如：

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=10, out_features=5) # 包含 weight 和 bias
        self.lone_parameter = nn.Parameter(torch.randn(3)) # 一个独立的参数
model = SimpleModel()
for param in model.parameters():
    print(f"参数形状: {param.shape}, 需要梯度: {param.requires_grad}")

# 输出:
# 参数形状: torch.Size([5, 10]), 需要梯度: True   <-- self.linear_layer.weight
# 参数形状: torch.Size([5]), 需要梯度: True      <-- self.linear_layer.bias
# 参数形状: torch.Size([3]), 需要梯度: True      <-- self.lone_parameter
```

**Parameter Groups 参数组**：除了使用上述的 parameter iterator 创建优化器外，`Pytorch`​ 还支持传入一组不同配置的参数（一个字典组成的列表），列表中每个字典定义了一个参数和对应的超参，包含 `params`​ （`torch.nn.Parameter`​ 对象的列表，而非迭代器）以及训练这些模型参数的超参（例如 `lr`​、`weight_decay`​ 等）。例如外面在前面例子的基础上希望对每层参数使用不同的学习率，则可以按下面的方式定义优化器

```python
weight_param = model.linear_layer.weight
bias_param = model.linear_layer.bias
other_param = model.lone_parameter

# 注意 params 键对应的值必须为列表
param_groups = [
    {'params': [weight_param], 'lr': 0.01},    # 权重组
    {'params': [bias_param], 'lr': 0.005},     # 偏置组
    {'params': [other_param], 'lr': 0.1}       # 独立参数组
]

optimizer = optim.SGD(param_groups) # 使用SGD举例
```

注意：`model.parameters()`​ 返回的只能返回 iterator。parameter groups 必须要在外部定义。

**model.named_parameters()** ：为了更方便地筛选和命名参数， `Pytorch`​ 还支持 `named_parameters()`​ 方法，类似于 `parameters()`​ 方法，其返回了一个迭代器，但是每步迭代得到的的元素是 `(name, parameters)`​ 的元组。那 `name`​ 是哪里来的呢？Pytorch 有一套自己的命名方式：

- 直接赋值：例如 `self.output_layer = nn.Linear(10,2)`​，则对应的两个参数的名称为 `output_layer.weight`​ 和 `output_layer.bias`​。再比如说 `self.my_param = nn.Parameter(torch.rand(5))`​，则对应的名称为 `my_param`​
- ​`nn.Sequential`​ 结构：例如 `self.features = nn.Sequential(xxx, xxx)`​，则获得的参数的名称为 `features.0.weight`​、`features.0.bias`​、`features.1.weight`​ ... 其中中间的数字表示第 `n`​ 层。
- ​`nn.ModuleList`​ 或者 Python 列表/字典：类似于 `nn.Sequential`​。

我们可以使用 `'pattern' in name`​ 来确定参数的 `name`​ 中是否包含某个 `pattern`​。例如我们也可以用 `name`​ 的方式来构建 `param groups`​：

```python
model = SimpleModel()

weights_params = [p for name, p in model.named_parameters() if "weight" in name]
bias_params = [p for name, p in model.named_parameters() if "bias" in name]
other_params = [p for name, p in model.named_parameters() if "bias" not in name and "weight" not in name]

param_groups = [
    {'params': weights_params, 'lr': 0.001},
    {'params': bias_params, 'lr': 0.01}, # 给 bias 设置 10 倍的学习率
    {'params': other_params, 'lr': 0.001, 'weight_decay': 0} # 独立参数不使用权重衰减
]

optimizer = torch.optim.AdamW(param_groups)
```

## Muon 优化器实战

现在我们拥有了所有必要的知识，可以来解读和使用 `MuonWithAuxAdam`​ 了（见 [Muon/muon.py at master · KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py)），从代码和文档我们可以看出：

- ​`Muon`​ 是一个混合优化器：其内部同时实现了 `Muon`​ 和 `Adam`​ 的更新逻辑；
- ​`Muon`​ 必须传入参数组（不能传入 `model.parameters()`​ 或者 `model.named_parameters()`​），参数组中使用键 `'use_muon': True/False`​ 来确定是否使用 Muon 算法。

我们再次来理解 `Muon`​ 官方仓库给出的范例

```python
from muon import MuonWithAuxAdam
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)
```

- ​`model.body`​、`model.head`​、`model.embed`​ 是模型中定义的属性，但这并非是所有模型都有这些属性，因此要根据自己的模型去修改。
- 这里把 `2`​ 维及以上的参数设置为 `hidden_weights`​ 使用 `Muon`​ 进行优化，`1`​ 维偏置以及 `head`​、`embed`​ 的参数使用 `Adam`​ 优化。
- 具体怎么控制是否使用 `muon`​：创建一个参数组，并使用 `use_muon`​ 键控制是否使用 `muon`​ 进行优化。

例如比较常用的，筛选掉维度 `>=2`​ 以及 `embedding`​ 层的用法可以是：

```python
# 筛选出所有维度 >= 2 的参数，通常是权重 (weights)
muon_params = [
    p for name, p in model.named_parameters() 
    if p.ndim >= 2 and 'embed' not in name
]

# 剩余的所有参数都交给 AdamW
# 包括：所有维度 < 2 的参数 (biases, layernorm gains) 以及 Embedding 层的参数
adam_params = [
    p for name, p in model.named_parameters() 
    if p.ndim < 2 or 'embed' in name
]

# 检查是否所有参数都被分配了
assert len(list(model.parameters())) == len(muon_params) + len(adam_params)

param_groups = [
    dict(params=muon_params, use_muon=True, lr=0.02, weight_decay=0.01),
    dict(params=adam_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]

optimizer = MuonWithAuxAdam(param_groups)
```

‍

‍
