+++

title = "Transformers 库 CLM 预训练与微调代码阅读"

date = "2025-08-19"

[taxonomies]

tags = ["Machine Learning", "Transformers Library"]

+++

Hugging Face `transformers`​ 库支持对 CLM（Causal Language Modeling）和 MLM（Masked Language Modeling）的微调与预训练，本次对 CLM 的预训练/微调代码进行阅读。

> 代码见 [transformers/examples/pytorch/language-modeling/run_clm.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)

## 参数定义 Argument Parsing

`run_clm.py`​ 定义了三个 `dataclass`​ 来管理所有可配置的参数。

**ModelArguments 模型参数**：定义了模型、配置、分词器相关的参数

- `model_name_or_path`​：指定要使用的预训练模型，例如 `gpt2`​，或者一个本地模型路径。如果为空，则表示要从头开始预训练模型；
- `model_type`​：如果从头训练，则需要指定模型类型，例如 `gpt2`​；
- `config_name`​、`tokenizer_name`​：配置和分词器的名词，一般不需要指定，脚本会自动从 `model_name_or_path`​ 中加载；
- `cache_dir`​：下载模型和数据集的缓存路径，默认为 `~/.cache/huggingface`​；
- `torch_dtype`​：加载模型时的数据类型，例如 `float16`​ 或 `bfloat16`​，用于混合精度模型；

**DataTrainingArguments 数据参数**：定义了数据集加载和预处理相关的参数

- `dataset_name`​：要从 Hugging Face Hub 加载的数据集名称，例如 `wikitext`​；
- `train_file`​、`validation_file`​：如果使用本地数据，则指定训练和验证文件的路径（支持 `.txt`​、`.csv`​、`.json`​）；
- `block_size`​：数据预处理后，所有文本会被拼接起来，然后切分成固定长度（`block_size`​）的块。这个值通常取决于模型的最大序列长度限制（如 GPT-2 是 1024）；
- `streaming`: 在**数据预处理阶段**是否启用流式处理模式。如果不开启 `streaming`​，则会预先将数据预处理完毕然后写入磁盘缓存，训练过程中直接从缓存中读取。如果开启 `streaming`​，则训练过程中是边处理数据边送给模型训练。
- `overwrite_cache`​：是否覆盖预处理后的缓存数据。

**TrainingArguments 训练参数**：包含了所有训练超参数

- `output_dir`​：训练结果（模型权重、checkpoint 等）的输出目录；
- `do_train`​, `do_eval`: 是否执行训练和评估；
- `per_device_train_batch_size`: 每个 GPU 的训练批次大小；
- `learning_rate`: 学习率；
- `num_train_epochs`: 训练的总轮数；
- `fp16`​, `bf16`: 是否启用16位浮点数（混合精度）训练；
- `push_to_hub`: 训练结束后是否将模型推送到 Hugging Face Hub。

## Main 函数逻辑

**Step 1. 解析参数**：`HfArgumentParser`​ 会解析命令行传入的参数，填充到之前定义的三个 `dataclass`​ 对象中。

```python
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

**Step 2. 日志、种子**：设置好 `logger`​、`seed`​、`output_dir`​。检查 `output_dir`​ 中是否存在训练到一半的 checkpoint，若存在则接着训练。

**Step 3. 加载数据集**：准备好数据集

- 判断来源：判断通过 `dataset_name`​ （在线 Hub）还是 `train_file`​ （本地文件）提供数据；
- 加载数据：使用 `datasets.load_dataset`​ 函数加载数据；
- 验证集处理：如果用户只提供了训练集而没有验证集，脚本会自动从训练集中切分出一部分（默认为 5%）作为验证集。对于流式数据集（streaming），使用了 `split_streaming_dataset`​ 函数，通过取模运算（`i % 100`​）来动态切分数据流。

**Step 4. 加载模型和分词器**：使用 `Auto`​ 系列的自动化类，用户只需要提供模型名称，其余都会自动处理

```python
config = AutoConfig.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
model = AutoModelForCausalLM.from_pretrained(...)
```

**Step 5. 模型预处理**：使用 `.map()`​ 方法将原始文本数据转换为模型输入格式，`.map`​支持多进程加速和缓存，效率很高

- Tokenization：定义 `tokenize_function`​函数，其输入一批文本，输出 token ID 列表；
- Grouping Texts：定义 `group_texts`​ 函数，其完成以下功能：

  - 将所有样本的 token ID 拼接为一个巨大的列表；
  - 将长列表切割为多个长度为 `block_size`​ 的小块；
  - 每个小块既是模型的 `input_ids`​，也是 `labels`​

**Step 6. 初始化 Trainer**：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    # ...
)
```

**Step 7. 执行训练与评估**：

```python
# 训练
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() # 保存最终模型
    trainer.save_state() # 保存训练状态
    trainer.log_metrics("train", metrics) # 记录训练指标
# 评估
if training_args.do_eval:
    metrics = trainer.evaluate()
    # 计算困惑度 Perplexity
    perplexity = math.exp(metrics["eval_loss"])
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
```
