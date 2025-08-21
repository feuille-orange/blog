+++

title = "KL 散度、交叉熵与对数似然"

date = "2025-08-21"

[taxonomies]

tags = ["Machine Learning", "Statistics"]

+++



## KL 散度

KL 散度（Kullback-Leibler Divergence）是一种非对称的度量，用于量化一个概率分布和另一个参考概率分布的差异。

**KL 散度的定义**：给定概率分布 $P(x)$ 和 $Q(x)$，若它们是离散概率分布，则 *KL 散度*定义为

$$
D_{KL} (P ||Q) := \sum_x P(x) \log \frac{P(x)}{Q(x)}.
$$

若 $P(x)$ 和 $Q(x)$ 是连续概率分布，对应概率密度函数为 $p(x)$ 和 $q(x)$，则 KL 散度为

$$
D_{KL} (P ||Q) := \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} \mathrm{d} x.
$$

**KL 散度的性质**：KL 散度满足除了对称性外的度量性质

- 非负性：$D_{KL}(P || Q) \geq 0$，且 $D_{KL}(P || Q) = 0$ 当且仅当 $P(x) \equiv Q(x)$。
- 非对称性：$D_{KL}(P || Q) \neq D_{KL}(Q || P)$。

**KL 散度的信息论解释**：$D_{KL}(P || Q)$ 表示使用 $Q$ 来编码 $P$ 时，平均每个样本所需的额外信息量（比特数）。KL 散度越小，说明 $Q$ 对 $P$ 的近似程度越好，因此**最小化 KL 散度**也是机器学习中常用的优化目标。

## 信息熵与交叉熵

信息熵（Information Entropy）用于衡量一个随机变量的不确定性，一个系统越混乱，越不可预测，则其信息熵越高。交叉熵（Cross Entropy）衡量着我们使用**错误的**分布 $Q$ 去衡量真实分布 $P$ 时所需要的平均编码长度。

**信息熵的定义**：信息熵定义为一个随机变量所包含信息的平均期望，给定离散随机变量 $X$，其可取 $x_1, x_2, \cdots, x_n$，对应概率 $P(x_1), P(x_2), \cdots, P(x_n)$，则*信息熵*定义为

$$
H(x) = - \sum_{i = 1}^n P(x_i) \log_bP(x_i),
$$

其中 $b$ 决定了熵的单位，$b = 2$ 表示 bit，$b=e$ 表示 nat，$b = 10$ 表示 hart。

**交叉熵的定义**：给定概率分布 $P(x)$ 和 $Q(x)$，*交叉熵*函数为

$$
H(P, Q) := - \sum_xP(x)\log Q(x).
$$

**交叉熵的信息论理解**：如果 $Q$ 能完美预测真实分布 $P$，那么交叉熵就等于真实分布的信息熵 $H(P)$，此时编码成本最低。如果 $Q$ 的预测非常不准，那么交叉熵就会远大于信息熵。因此在机器学习中我们往往希望**最小化交叉熵**。

## 对数似然

似然函数在统计学中用于**评估模型参数对观测数据的拟合程度**，似然函数越大说明当前模型的参数对观测数据的拟合程度越高。

**似然函数 Likelihood Function**：给定数据集 $X = \{x_1, x_2, \cdots, x_n\}$ 和由参数 $\theta$ 控制的概率模型 $P(X|\theta)$，*似然函数*定义为：

$$
L(\theta|X) := P(X|\theta) = \prod_{i = 1}^n P(x_i|\theta).
$$

即表示着在 $\theta$ 参数情况下，从概率模型中抽取得到数据集 $X$ 的概率。

**最大化似然估计 MLE**：显然 $L(\theta|X)$ 越大，那么模型就越可能生成当前的数据集，也就是更加贴合数据集。因此一般我们的目标是寻找一组参数 $\hat{\theta}$，使得似然函数最大化

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta|X).
$$

**对数似然函数 Log-Likelihood Function**：由于似然函数的格式是连乘，在计算上既复杂又容易数值下溢，因此通常会使用*对数似然函数 Log-Likelihood Function*：

$$
\log L(\theta|X) = \sum_{i = 1}^n \log P(x_i|\theta).
$$

可以看出最大化似然估计也等价于最大化对数似然估计，因此实际优化中一般我们都使用最大对数似然。

## 三者的关系

**KL 散度与交叉熵**：KL 散度为交叉熵与真实分布熵的差，即

$$
D_{KL}(P||Q) = H(P,Q) - H(P).
$$

因此**最小化 KL 散度等价于最小化 Cross-Entropy**，而在计算时由于交叉熵更容易计算，因此通常使用交叉熵函数。

**KL 散度与对数似然**：这里我们假设数据集由数据分布 $P_{\text{data}}(x)$ 生成，分布模型为 $P_{\text{model}}(x|\theta)$，此时

$$
D_{KL}(P_{\text{data}} || P_{\text{model}}) = \sum_{x} P_{\text{data}}(x) \log P_{\text{data}}(x) - \sum_{x} P_{\text{data}}(x)\log P_{\text{model}}(x)
$$

右侧第一项即 $P_{\text{data}}$ 的信息熵，由于 $P_{\text{data}}$ 是固定的，因此此项是一个常数，而第二项即 $P_{\text{data}}$ 和 $P_{\text{model}}$ 的交叉熵。因此**最小化 KL 散度等价于最大化对数似然**。

‍
