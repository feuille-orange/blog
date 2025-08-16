+++
title = "Hugging Face trl 微调库 SFT 代码阅读"
date = "2025-08-16"

[taxonomies]
tags = ["Fine Tuning", "Machine Learning", "TRL Library"]
‍+++

## SFT 代码阅读

> 代码见 [trl/trl/scripts/sft.py at main · huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)

**Step 1. 模型和分词器初始化**：

```python
# 检查是否要对模型进行量化，减少显存占用
quantization_config = get_quantization_config(model_args)
# 准备模型的参数：例如 attn_implementation 使用哪种注意力机制
model_kwargs = dict(...)
# 加载 Hugging Face 预训练好的模型
model = AutoModelForCausalLM.from_pretrained(...)
# 加载 Hugging Face 预训练好的分词器
tokenizer = AutoTokenizer.from_pretrained(...)
```

**Step 2. 数据集加载**：可以用 `--dataset_name`​ 加载一个标准数据集，也可以用 `--datasets`​ 提供更复杂的配置（例如混合多个不同的数据集）

```python
if dataset_args.datasets and script_args.dataset_name:
    # ...
elif ...:
    dataset = get_dataset(dataset_args)
elif ...:
    dataset = load_dataset(...)
```

**Step 3. SFT 训练器初始化**：

```python
trainer = SFTTrainer(
    model=model, # 模型
    args=training_args, # 训练配置
    train_dataset=dataset[...], # 训练数据集
    eval_dataset=dataset[...], # 验证数据集
    tokenizer=tokenizer, # 分词器
    peft_config=get_peft_config(model_args), # PEFT 配置，如果用 LoRA，则会生成，否则为 None
)
```

**Step 4. 训练与保存**：

```python
# Train the model
trainer.train()

# Save and push to Hub
trainer.save_model(training_args.output_dir)
if training_args.push_to_hub:
    trainer.push_to_hub(...)
```

‍
