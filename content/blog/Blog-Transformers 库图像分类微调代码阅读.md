+++
title = "Transformers 库图像分类微调代码阅读"
date = "2025-08-12"

[taxonomies]
tags = ["Machine Learning", "Transformers Library", "Fine Tuning", "Computer Vision"]
+++

近期想做一些图像大模型微调的测试，找到了 Hugging Face 的 transformers 仓库中有一些现成的脚本可以进行微调，因此在此详细阅读和解析下使用的代码。具体代码见 [transformers/examples/pytorch/image-classification at main · huggingface/transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)。

## 参数定义

脚本使用 `@dataclass`​ 装饰器（用法可以问 AI）定义了三个类来管理所有的参数：

- ​`DataTrainingArguments`​：控制数据相关参数，例如 `dataset_name`​、`train_dir`​、`validation_dir`​ 等；
- ​`ModelArguments`​：模型参数，最重要的是 `model_name_or_path`​ 指定了要使用的预训练模型，例如 `"google/vit-base-patch16-224-in21k"`​
- ​`TrainingArguments`​：训练过程相关参数，例如 `output_dir`​、`learning rate`​、`num_train_epochs`​、`per_device_train_batch_size`​ 等。

## Main 函数逻辑

**Step 1. 初始化与设置**：解析参数、设置日志、检查断点、设置随机种子（此处略去详细代码）

**Step 2. 加载数据集**：

- 如果提供了 `dataset_name`​ 则从 Hugging Face 自动下载和加载数据集，生成一个 `DatasetDict`​ 对象

```python
if data_args.dataset_name is not None:
    dataset = load_dataset(
        data_args.dataset_name,
        ...
    )

# dataset 对象的结构示意：
# 其中 features 列要去 hugging face 上看对应的属性，不一定就是 image 和 label
# DatasetDict({
#    'train': Dataset({
#        features: ['image', 'label'],
#        num_rows: 50000
#    }),
#    'test': Dataset({
#        features: ['image', 'label'],
#        num_rows: 10000
#    })
# })

```

- 如果没提供 `dataset_name`​，则需要提供本地数据集的 `train_dir`​ 和 `validation_dir`​。使用 `imagefolder`​ 这个特殊的加载器，自动将**子文件夹的名称作为类别标签**。

```python
else:
    data_files = {} # data_files 是一个字典，有 train 和 validation 两个 key
    if data_args.train_dir is not None:
        data_files["train"] = os.path.join(data_args.train_dir, "**")
    if data_args.validation_dir is not None:
        data_files["validation"] = os.path.join(data_args.validation_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )
```

- 准备 Label Mappings：数据集中的标签是类似于 cat、dog 等单词，但是模型只能理解数字。因此我们要准备 `label2id`​ 和 `id2label`​ 两个映射

```python
labels = dataset["train"].features[data_args.label_column_name].names
label2id, id2label = {}, {}
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
```

**Step 3. 加载模型和处理器**：加载模型配置文件、加载模型权重、加载模型的图像处理器

```python
# AutoConfig 首先加载模型原始配置文件 config.json，然后用我们自己的数据集更新它
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
    # ...
)

# AutoModelForImageClassification 下载并加载模型的预训练权重，并传入 config
# ignore_mismatched_sizes 表示原始模型的 head 和新模型 head 不匹配，并表示这是正常的
model = AutoModelForImageClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    ignore_mismatched_sizes=True, # 非常重要！
    # ...
)

# AutoImageProcessor 表示加载与模型配套的图像处理器，从 preprocessor_config.json 读取
# 读取信息包括 image_mean、image_std 等
image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path, ...)
```

**Step 4. 图像变换与数据增强**：将原始的 `PIL.Image`​ 对象转换为 `torch.Tensor`​

```python
# 核心代码 (以训练集为例)
# _train_transforms 是训练图像的处理管道，输入一张图片，输出一个 torch.Tensor
_train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

# 批处理函数，接受一个批次的数据，输出时增加一个 pixel_values 的 key，存储对应的 torch.Tensor
# 注意原 img/image 的 Key 依旧保留
def train_transforms(example_batch):
    example_batch["pixel_values"] = [
        _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
    ]
    return example_batch

# set_transform 表示不会立刻对数据集执行 Transform，而是 Trainer 需要一批数据，再做 Transform
dataset["train"].set_transform(train_transforms)
```

**Step 5. 设置训练前 Trainer**

- ​`compute_metrics`​ 函数

```python
# p 是一个 EvalPrediction 对象，评估阶段，Trainer 将模型的预测结果和真实标签打包为一个 EvalPrediction 对象
def compute_metrics(p): 
    # 每一行（每个样本的预测分数）取最大值的索引，即模型的预测 ID
    predictions = np.argmax(p.predictions, axis=1)
	# 将预测的 ID 列表和真实 ID 列表传递给 evaluate 库的 accuracy 指标，自动计算准确率
    return metric.compute(predictions=predictions, references=p.label_ids)
```

- ​`collate_fn`​ 函数：输入一个批次的数据，将这些数据整理为 `model.forward()`​ 所需的数据格式

```python
# examples 是 DataLoader 取出的一个批次的样本
def collate_fn(examples):
	# 取出 pixel_values 张量，拼成一整个张量
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
	# 去除 label_column_name 张量，拼成一个张量
    labels = torch.tensor([example[data_args.label_column_name] for example in examples])
    # 返回一个字典，字典的 Key 是 pixel_values 和 labels，value 为对应的 Tensor
	return {"pixel_values": pixel_values, "labels": labels}

# examples 是一个列表，每个元素都是经过 train_transforms 处理后的样本
# [
#  {'pixel_values': <Tensor_1>, 'label': 0},
#  {'pixel_values': <Tensor_2>, 'label': 5},
#  {'pixel_values': <Tensor_3>, 'label': 2},
#  {'pixel_values': <Tensor_4>, 'label': 0}
# ]

# 最终输出一个字典，满足 model.forward() 方法期望的输入格式
# {"pixel_values": <Tensor BxCxHxW>, "labels": <Tensor B>}
```

‍

‍

‍
