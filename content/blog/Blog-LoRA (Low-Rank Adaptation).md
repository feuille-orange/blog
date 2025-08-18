+++

title = "LoRA (Low-Rank Adaptation)"

date = "2025-08-18"

[taxonomies]

tags = ["Fine Tuning", "Machine Learning"]

+++



## 背景：全微调（Full Fine-Tuning）

在 LoRA 出现前，一般进行大模型微调都会执行全微调，即将原模型的所有参数都在下游任务上进行调整。

**网络层的数学表示**：这里以线性层（全连接层）为例，假设输入向量为 $x \in \mathbb{R}^k$，输出向量为 $h \in \mathbb{R}^d$，该层的计算可以表示为：

$$
h = W x,
$$

其中 $W \in \mathbb{R}^{d \times k}$ 是该层权重矩阵。

**全量微调**：假设预训练好的权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$，*Full Fine-Tuning* 在新的数据集上更新完整的权重矩阵

$$
W = W_0 + \Delta W,
$$

其中 $\Delta W \in \mathbb{R}^{d \times k}$ 是微调过程中的权重更新量。

## LoRA 的思想假设

**低秩假设**：LoRA (Low-Rank Adaptation) 的基本思想是微调过程中，权重更新矩阵 $\Delta W$ 有很低的 intrinsic rank，即 $\Delta W$ 尺寸虽然很大，但是其包含的信息可以用远小于其维数的秩 $r$ 来有效表示。

**低秩分解 Low-Rank Decomposition**：LoRA 不学习巨大的 $\Delta W$ 矩阵本身，而是通过 *Low-Rank Decomposition* 来近似它，即

$$
\Delta W = BA,
$$

其中 $B \in \mathbb{R}^{d \times r}$， $A \in \mathbb{R}^{r \times k}$，$0 < r \ll \min\{d, k\}$ 为 LoRA 近似矩阵的秩。通常 $A$ 使用高斯分布初始化，$B$ 使用零初始化，这样 $\Delta W$ 就被初始化为 $\mathcal{O}$。

**LoRA 的更新步骤**：在实际程序中，我们一般会见到 `lora_r`​ 和 `lora_alpha`​ 两个超参数，因为实际更新的方式为

$$
W = W_0 + \frac{\alpha}{r}(BA),
$$

其中 $r$ 是 LoRA rank，$\alpha$ 为 scaling factor。一般来说，我们会将 `lora_alpha`​ 设置得比 `lora_r`​ 大一点，例如 $\alpha = 2r$​。

**LoRA 训练与推理**：在训练过程中，我们冻结 $W_0$，只更新 $A, B$。而在推理部署时，将 $A,B$ 合并回权重矩阵 $W$，这样我们在推理过程中也不需要额外的内存/显存开销。

‍
