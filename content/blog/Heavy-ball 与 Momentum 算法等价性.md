+++

title = "Heavy-ball 与 Momentum 算法等价性"

date = "2025-08-16"

[taxonomies]

tags = ["Optimization"]

+++

依稀记得 Heavy-Ball 动量格式和 Momentum 动量格式是完全等价的，但是总是记不清楚，因此重新写一遍。

**Heavy-Ball 动量格式**：Heavy-Ball 动量格式为

$$
\theta_{t+1} - \theta_t = \gamma (\theta_{t} - \theta_{t-1}) - \eta \nabla f(\theta_t)
$$

**现代 Momentum 格式**：现代 Momentum 动量格式为

$$
v_{t+1} = \beta v_t + \nabla f(\theta_t), \quad \theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

**Heavy-Ball 与 Momentum 格式的等价性**：Heavy-Ball 和 Momentum 在数学上是完全等价的，且 $\gamma = \beta$。

> 证明：这里只从 Momentum 出发推导 Heavy-Ball，将 Momentum 第一个式子两侧同时乘上学习率 $\eta$ 得到
>
> $$
> \eta v_{t+1} = \beta \eta v_t + \eta \nabla f(\theta_t),
> $$
>
> 而根据第二个式子得到 $-\eta v_{t+1} = \theta_{t+1} - \theta_t$，因此带入上面的等式得到结论。

‍
