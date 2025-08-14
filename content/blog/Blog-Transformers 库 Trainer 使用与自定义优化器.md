+++

title = "Transformers 库 Trainer 使用与自定义优化器"

date = "2025-08-13"

[taxonomies]

tags = ["Machine Learning", "Transformers Library"]

+++

近期想使用 `Hugging Face`​ 的 `Trainer`​ 测试一些微调和预训练任务，因此学习一下如何使用 `Trainer`​ 以及如何在其中自定义优化器。

## Hugging Face Trainer 原理

**Trainer 的目的**：虽然不用 `Trainer`​，`Pytorch`​ 的训练流程看起来也不麻烦，但是如果我们需要加入**分布式训练**、**混合精度训练**、**梯度累积**、**日志记录**、**评估与断点续训**时，代码量和复杂度会快速上升。而 `Trainer`​ 的目的就是让用户从这些复杂的细节中解放出来。

**Trainer 的创建**：`Trainer`​ 的创建 API 主要如下

```python
trainer = Trainer(
    model: PreTrainedModel, # 要训练的 Hugging Face 模型实例，继承自 PreTrainedModel
    args: TrainingArguments, # 训练超参数和选项
    train_dataset: Optional[Dataset] = None, # 训练数据集，datasets.Dataset 对象
    eval_dataset: Optional[Dataset] = None, # 评估数据集
    data_collator: Optional[DataCollator] = None, # 将 list of samples 转为一整个 Tensor
    compute_metrics: Optional[Callable] = None,  # 自定义函数，用于在评估过程中计算性能指标
    tokenizer: Optional[PreTrainedTokenizerBase] = None, 
    callbacks: Optional[List[TrainerCallback]] = None, # 回调系统，可以在训练周期特定节点（例如 on_train_begin）执行自定义代码
    optimizers: Tuple[Optimizer, Scheduler] = (None, None), # 自定义优化器和 scheduler。
    ...
)
```

**Trainer 的主要方法**：

- `train(resume_from_checkpoint: str = None)`​：启动训练，允许从某个 `checkpoint`​ 路径恢复训练。
- `evaluate(eval_dataset: Dataset = None)`​：在给定的数据集上执行一次评估。
- `predict(test_dataset: Dataset)`​：对一个没有标签的测试集进行预测，返回原始输出。
- `save_model(output_dir: str=None)`​：将模型、配置、分词器（如果提供了）保存到指定目录
- `push_to_hub()`​：将训练好的模型、分词器等推送到 Hugging Face Hub。

**Trainer 的训练过程**：对于一般用户只需要调用 `trainer.train()`​，其内部会执行一套高度优化、功能完备的训练循环。关键部分如下：

- 环境与设置初始化：读取传入的 `TrainingArguments`​ 的配置，利用 `Accelerate`​ 库探测硬件环境；
- 数据加载器的准备：读取传入的 `Dataset`​ 对象，创建 `DataLoader`​，使用 `DistributedSampler`​ 确保每个 GPU 进程拿到数据的不重复子集；
- 核心训练循环 `training_step`​：每个训练步骤中取出一个 Batch 的数据传递给 `training_step`​。其中负责将数据移动到 `device`​，调用 `model(**input)`​ 执行前向传播，并取出 `loss`​；
- 优化过程：使用 `GradScaler`​ 防止梯度下溢，控制 `loss.backward()`​、`optimizer.step()`​、`optimizer.zero_grad()`​ 累计梯度。
- 生命周期钩子 Lifecycle Hooks：`Trainer`​ 提供了一系列回调函数，允许在特定节点（如 `on_train_begin`​、`on_step_end`​、`on_epoch_end`​、`on_save`​ 等）注入自定义逻辑，例如提前停止、打印自定义日志等。

## Hugging Face Trainer 自定义优化器

**使用 Trainer 支持的优化器**：​`Trainer`​ 默认支持 `AdamW`​、`Adafactor`​ 等优化器，可以通过 `--optim`​ 参数选择。

**使用 Trainer 不支持的优化器**：最方便的方式是继承 `Trainer`​ 并重写 `create_optimizer`​ 方法。具体而言，`Trainer`​ 在初始化过程中会调用 `self.create_optimizer_and_scheduler()`​，其内部再调用 `self.create_optimizer()`​。

- 创建自定义的 `TrainingArguments`​

```python
@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    自定义训练参数，增加了优化器选项及其相关超参数。
    """
    optim_name: Optional[str] = field(
        default="adamw",
        metadata={"help": "要使用的优化器 (可选项: adamw, adan, muon, pid_adamw)"}
    )
    ki: float = field(
        default=0.0, 
        metadata={"help": "PIDAdamW 的积分项 (I)"}
    )
    kp: float = field(
        default=0.0, 
        metadata={"help": "PIDAdamW 的比例项 (P)"}
    )
    kd: float = field(
        default=0.0, 
        metadata={"help": "PIDAdamW 的微分项 (D)"}
    )

# 在 main 的 HfArgumentParser 中修改为自定义的 TrainingArguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
```

- 创建自定义 `Trainer`​ 并重写 `create_optimizer`​

```python
from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def create_optimizer(self):
        """
        Create customed optimizers
        """
        opt_name = self.args.optim_name.lower()
        parameters = self.model.parameters()
        if opt_name == 'pidadamw':
          self.optimizer = PIDAdamW_AdSI(parameters, lr=self.args.learning_rate, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay, ki=self.args.ki, kd=self.args.kd)
        elif opt_name == 'adan':
          self.optimizer = Adan(parameters, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif opt_name == 'adamw':
          logger.info("Using default AdamW optimizer via super().create_optimizer()")
          return super().create_optimizer() # 调用父类
        else:
          logger.warning(f"Unsupported optimizer '{opt_name}' in custom logic. Falling back to default Trainer implementation.")
          return super().create_optimizer()

        return self.optimizer
```

- 在训练脚本中使用 `CustomTrainer`​

```python
# ... 在你的主训练脚本中 ...

# model, data_args, training_args 等都已准备好

# 不再使用：trainer = Trainer(...)
# 而是使用：
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 后续调用完全不变
trainer.train()
```
