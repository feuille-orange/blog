+++

title = "Fisher 矩阵与弹性权重巩固 Elastic Weight Consolidation, EWC"

date = "2025-09-01"

[taxonomies]

tags = ["Math", "Statistics"]

+++

‍

---

## 预备知识：似然函数

**似然函数**：似然函数 $L(\theta | D)$ 表示在参数为 $\theta$ 的模型下，观测到数据 $D$ 的概率，即

$$
L(\theta|D)=p(D|\theta)=\prod_{i=1}^Np(x_i|\theta).
$$

**对数似然**：由于似然函数的计算是乘法，有各种各样的问题。我们希望将其转换为加法或者减法，而取对数是一种很自然的想法

$$
\mathcal{L}(\theta|D)=\log L(\theta|D)=\sum_{i=1}^N\log p(x_i|\theta).
$$

**最大化似然估计 Maximum Likelihood Estimation, MLE**：如果我们能够调整 $\theta$，那么我们肯定希望在 $\theta$ 下观测到 $D$ 的概率越高越好，也就是最大化似然函数

$$
\theta_{MLE}=\arg\max_\theta\mathcal{L}(\theta|D).
$$

---

## 预备知识：Score 函数

**Score 函数的定义**：定义 Score 函数为对数似然函数关于 $\theta$ 的梯度，其表示着在<u>当前 </u>​<u>$\theta$</u>​<u> 位置如何调整参数能最快地提高模型对数据的解释能力</u>。

$$
s(\theta) := \nabla_\theta \mathcal{L}(\theta|D)
$$

**Score 函数的期望**：假设数据 $D$ 是由一个真实的参数 $\theta^\ast$ 生成的，那么 Score 函数在该点的期望为零：

$$
\mathbb{E}_{D\sim p(D|\theta^*)}[s(\theta^*)]=0,
$$

需要注意的是，这个期望为 $0$ 是一个关于所有可能数据集的<u>理论平均性质</u>。对于我们手中任何一个具体的数据集 $D$，由于采样的随机性，计算出的 $s(\theta^*)$ 的值几乎总是不为 $0$ 的。

---

## Fisher 矩阵

**Fisher 信息矩阵**：Fisher 信息矩阵量化了<u>数据 </u>​<u>$D$</u>​<u> 中包含了多少模型参数 </u>​<u>$\theta$</u>​<u> 的信息</u>，其有两种等价的定义。

**定义 1. Score 函数的方差**：定义 Fisher 信息矩阵为 Score 函数的协方差矩阵

$$
F(\theta)=\mathbb{E}_{x\sim p(x|\theta)}[s(\theta)s(\theta)^\top]
$$

在实际问题中，对于每个样本的梯度向量 $s_k(\theta)$，我们通过以下公式计算 Fisher 信息矩阵

$$
\hat{F}(\theta)=\frac{1}{N}\sum_{k=1}^Ns_k(\theta)s_k(\theta)^\top
$$

**定义 2. 对数似然函数 Hessian 矩阵的负期望**：我们也可以用对数似然函数的 Hessian 矩阵的负期望来定义 Fisher 矩阵

$$
F(\theta)=-\mathbb{E}_{x\sim p(x|\theta)}[\nabla_{\theta}^{2}\mathcal{L}(\theta|D)]
$$

在实际计算中，对数据集中每个样本 $x_k$ 计算对数似然的 Hessian 矩阵 $\nabla_\theta^2\mathcal{L}(\theta|x_k)=\nabla_\theta^2\log p(x_k|\theta)$，取负号后求和：

$$
\hat{F}(\theta)=-\frac{1}{N}\sum_{k=1}^{N}\nabla_{\theta}^{2}\log p(x_{k}|\theta)
$$

**Fisher 矩阵元素含义**：Fisher 矩阵的对角元素 $F_{ii}$ 衡量了数据中关于某个参数 $\theta_i$ 的信息量，$F_{ii}$ 越大说明 $\theta_i$ 对模型越重要。非对角元素 $F_{ij}$ 表示了不同参数 $\theta_i$ 和 $\theta_j$ 估计值之间的相关性/耦合度。

---

## 弹性权重巩固 Elastic Weight Consolidation, EWC

EWC 是一种持续学习算法，其目标是在学习新任务（任务B）时，减缓对旧任务（任务A）知识的遗忘。这里介绍其数学基础与基本想法。

**EWC 核心思想**：对于旧任务（任务A）中越重要的参数，在学习新任务（任务B）时就越要<u>限制它的改动</u>。

**EWC 损失函数**：假设已经完成了任务 A 的训练得到了最优参数 $\theta_A^\ast$。现在要学习任务 B，则 EWC 的损失函数表示为

$$
L(\theta)=L_B(\theta)+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_{A,i}^*)^2,
$$

其中 $L_B(\theta)$ 是任务 B 的标准损失函数，$\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_{A,i}^*)^2$ 是 EWC 增加的<u>正则化惩罚项</u>，$\lambda$ 是正则化参数，$(\theta_i-\theta_{A,i}^*)^2$ 是当前参数 $\theta_i$ 与任务 A 最优参数 $\theta_{A,i}^\ast$ 的二次距离，$F_i$ 是<u>在 </u>​<u>$\theta_{A}^\ast$</u>​<u> 处的 Fisher 矩阵</u>的对角元素，充当了<u>重要性权重</u>。

- 如果参数 $\theta_i$ 对任务 A 很重要（$F_i$ 很大），那么对它的任何改动 $(\theta_i - \theta_{A,i})^2$ 都会被放大，从而产生巨大的惩罚，迫使模型不要轻易改动它。
- 如果参数 $\theta_i$ 对任务 A 不重要（$F_i$ 很小），惩罚就小，模型可以自由调整它来适应任务 $B$。

**一些疑问和注意点**：

- 为什么只用对角元素：由于完整的 Fisher 矩阵过于庞大，EWC 中<u>只使用了 Fisher 矩阵的对角元素</u>以表示重要性，忽略了参数之间的相关性。
- Fisher 矩阵是固定的吗：注意 EWC 使用的 Fisher 矩阵是 $\theta_A^\ast$ 处的，而<u>非每次参数更新时更新</u>的，这样能准确捕捉对 A 任务重要的参数。

- 为什么用 Fisher 矩阵对角而非 Score 函数分量：理想情况下 $\theta_A^\ast$ 处的 Score 函数为 $0$（极值点），而 Fisher 矩阵衡量了<u>不同样本对某个参数的需求</u>，如果 $F_i$ 很大，则稍微一改动，便会影响多个样本的拟合结果。

---

‍
