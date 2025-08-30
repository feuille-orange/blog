+++

title = "Method of Successive Approximations 通用格式"

date = "2025-08-30"

[taxonomies]

tags = ["Optimal Control"]

+++

前面阅读学习了 MSA 算法求解 Optimal Control 问题，但是 MSA 的具体格式是基于神经网络结构的，本次总结了对于更加一般的 Optimal Control 问题，如何使用 MSA 进行求解。

---

## Optimal Control 问题描述

**Optimal Control 问题**：这里我们考虑动力系统

$$
\dot{x}(t) = f(t, x(t), u(t)),
$$

其中 $t \in [t_0, t_f]$ 是时间，$x(t) \in \mathbb{R}^n$ 是状态向量 state vector，$u(t) \in \mathcal{U} \subset \mathbb{R}^m$ 是控制向量，初值条件 $x(t_0) = x_0$，不考虑终值条件（如果有的话 Co-State Equation 的终值条件会有点儿不同）。Cost Functional 为

$$
J[u(\cdot)] = \phi(t_f, x(t_f)) + \int_{t_0}^{t_f} L(t, x(t), u(t))\mathrm{d}t.
$$

其中 $\phi(t_f, x(t_f))$ 是 terminal cost，$L(t, x(t), u(t))$ 是 running cost。

**PMP 格式**：考虑 Hamiltonian 函数：

$$
H(t,x(t),p(t),u(t)):=L(t,x(t),u(t))+p(t)^\top f(t,x(t),u(t))
$$

其中 $p(t)$ 是 co-state vector。PMP 必要条件包含：

- State Equation：如下方程满足初值条件 $x^\ast(t_0) = x_0$​

$$
\dot{x}^*(t)=\nabla_pH(t,x^*(t),p^*(t),u^*(t))=f(t,x^*(t),u^*(t))
$$

- Co-State Equation：如下方程满足终值条件

$$
\dot{p}^*(t)=-\nabla_xH(t,x^*(t),p^*(t),u^*(t))=-\left(\nabla_xL+(\nabla_xf)^\top p^*(t)\right)
$$

$$
p^*(t_f)=\nabla_x\phi(t_f,x^*(t_f))
$$

- Optimality Condition：

$$
u^*(t)=\arg\min_{u\in \mathcal{U}}H(t,x^*(t),p^*(t),u)
$$

---

## 通用 MSA 算法

- 初始化：给定初始控制函数 $u^{(0)}(t)$，设定迭代次数 $K$，初始化计数器 $k = 0$

- 求解 State Equation：给定当前控制函数 $u^{(k)}(t)$，从 $t_0$ 到 $t_f$ 正向求解 state equation

$$
\dot{x}^{(k)}(t)=f(t,x^{(k)}(t),u^{(k)}(t)), \quad x^{(k)}(t_0) = x_0.
$$

- 求解 Co-State Equation：利用 $x^{(k)}(t)$ 和控制 $u^{(k)}(t)$，从 $t_f$ 到 $t_0$ 反向计算 co-state equation

$$
\dot{p}^{(k)}(t)=-\nabla_xL(t,x^{(k)}(t),u^{(k)}(t))-\left[\nabla_xf(t,x^{(k)}(t),u^{(k)}(t))\right]^Tp^{(k)}(t),
$$

$$
p^{(k)}(t_f)=\nabla_x\phi(t_f,x^{(k)}(t_f)).
$$

- 更新控制函数：利用 $x^{(k)}(t)$ 和 $p^{(k)}(t)$ 更新 $u^{(k+1)}(t)$，对于每个时间点 $t \in [t_0, t_f]$，计算

$$
u^{(k+1)}(t)=\arg\min_{u\in \mathcal{U}}H(t,x^{(k)}(t),p^{(k)}(t),u)
$$

- 设置 $k = k+1$ 并且反复上述三步，直到达到迭代次数上限 $K$ 或者 $u^{(k)}(t)$ 的变化足够小：

$$
\int_{t_0}^{t_f}||u^{(k+1)}(t)-u^{(k)}(t)||^2dt<\epsilon.
$$

---

‍
