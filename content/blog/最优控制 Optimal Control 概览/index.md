+++

title = "最优控制 Optimal Control 概览"

date = "2025-08-15"

[taxonomies]

tags = ["Optimal Control"]

+++

近期在学习一些最优控制 Optimal Control 相关的理论，想先学习一下大致的问题，因此整理此笔记。

---

## 问题描述 Problem Formulation

**状态方程 State Equation**：Optimal Control 中考虑描述系统动态的常微分方程

$$
\dot{x}(t) = f(t, x(t), u(t)),
$$

其中 $t \in [t_0, t_f]$ 是时间，$x(t) \in \mathbb{R}^n$ 是状态向量 state vector，$u(t) \in \mathcal{U} \subset \mathbb{R}^m$ 是控制向量，$\mathcal{U}$ 是 set of admissible controls。

> 我们希望通过调整 $u$ 的控制信号来控制 $x(t)$ 的行为。

**边界条件 Boundary Conditions**：往往我们考虑的问题会有一些限制条件，例如

- Initial Condition：初始状态往往是固定的 $x(t_0) = x_0$；
- Terminal Condition：一般是一种约束条件 $\Psi(t_f, x(t_f)) = \mathbf{0}$，其中 $\Psi$ 是一个向量函数

**Cost Functional**：通过控制动力系统，我们希望最小化目标函数 $J$，其一般由两个部分组成：

$$
J[u(\cdot)] = \phi(t_f, x(t_f)) + \int_{t_0}^{t_f} L(t, x(t), u(t))\mathrm{d}t.
$$

- $\phi(t_f, x(t_f))$ 是 terminal cost，与系统在终点的状态有关；
- $L(t, x(t), u(t))$ 是 running cost，表示整个过程中的积累成本。

**我们的目标 Objective**：找到一个控制函数 $u^\ast(\cdot): [t_0, t_f] \to \mathcal{U}$，是的对应的状态轨迹 trajectory 能够最小化指标 $J$。

---

## Pontryagin's Minimum Principle - PMP

PMP 提供了一组最优控制所必须满足的必要条件，其是经典变分法 Calculus of Variations 的推广。但我们这里暂时不管什么是经典变分法，先把 PMP 的定理叙述搞明白。

**哈密顿量 The Hamiltonian**：我们引入一个辅助函数，称为*哈密顿量 Hamiltonian*：

$$
H(t,x,u,\lambda) = L(t, x, u) + \lambda(t)^\top f(t, x, u),
$$

其中 $\lambda(t) \in \mathbb{R}^n$ 是一个向量函数，称为 co-state vector 或 adjoint vector。哈密顿量将用于描述 PMP 定理。

**PMP 定理**：设 $u^\ast(\cdot)$ 是最优控制，$x^\ast(\cdot)$ 是对应的最优轨迹，那么必然存在一个非零的 co-state vector $\lambda^\ast(\cdot)$，对于所有 $t \in [t_0, t_f]$ 满足

- 状态方程 State Equation：系统原始动态的重新表述

$$
\dot{x}^\ast(t) = \frac{\partial H}{\partial \lambda}(t, x^\ast, u^\ast, \lambda^\ast) = f(t, x^\ast, u^\ast).
$$

- 协态方程 Co-state Equation/Adjoint Equation：$\lambda^\ast(t)$ 需要满足的方程

$$
\dot{\lambda}^\ast(t) = - \frac{\partial H}{\partial x}(t, x^\ast, u^\ast, \lambda^\ast).
$$

- 哈密顿量最小化条件 Minimization of the Hamiltonian：对于所有容许的控制 $v \in \mathcal{U}$，最优控制 $u^\ast(t)$ 必须使哈密顿量在任意时刻 $t$ 取最小值

$$
H(t, x^\ast(t), u^\ast(t), \lambda^\ast(t)) \leq H(t, x^\ast(t), v, \lambda^\ast(t)).
$$

> 如果控制 $u$ 没有约束（即 $\mathcal{U} = \mathbb{R}^m$）且 $H$ 对 $u$ 可微，则该条件可以简化为
>
> $$
> \frac{\partial H}{\partial u} (t, x^\ast, u^\ast, \lambda^\ast) = 0
> $$

- 横截条件 Transversality Condition：如果 $x(t_f)$ 固定，则 $\lambda(t_f)$ 没有约束；如果 $x(t_f)$ 自由，则

$$
\lambda^\ast(t_f) = \frac{\partial \phi}{\partial x} (t_f, x^\ast(t_f)).
$$

**总结**：PMP 将复杂的泛函最小化问题转化为两点边值问题 Two-Point Boundary Value Problem

---

## 动态规划与 Hamilton-Jacobi-Bellman (HJB) 方程

动态规划是解决最优控制问题的另一种方法，其提供了最优性的**充分条件**，并且能得到**闭环 closed-loop** 或者**反馈 feedback** 控制策略。

**最优值函数 Optimal Value Function**：定义 *Optimal Value Function* $V(t, x)$ 为从时间 $t$、状态 $x$ 出发，能得到的最小成本

$$
V(t, x) = \min_{u(\cdot) \in \mathcal{U}} \left[\phi(t_f, x(t_f)) + \int_t^{t_f} L(\tau, x(\tau), u(\tau) \mathrm{d} \tau\right],
$$

其中轨迹 $x(\tau)$ 满足 $\dot{x} = f(\tau, x, u)$ 和 $x(t) = x$。根据其定义，我们有 terminal condition

$$
V(t_f, x) = \phi(t_f, x).
$$

**Bellman's Principle of Optimality**：无论过去的状态和决策如何，余下的决策对于由过去决策所形成的状态而言，也必须构成一个最优策略。

**HJB 方程**：根据 Bellman's Priciple of Optimality，考虑一个极小的时间段 $[t, t+\delta t]$，

$$
V(t, x) = \min_{u(t)} [L(t,x,u) \delta t + V(t + \delta t, x(t + \delta t)) + o(\delta t)]
$$

对 $V(t + \delta t, x(t+\delta t))$ 进行泰勒展开

$$
V(t + \delta t, x+ \delta x) \approx V(t,x) + \frac{\partial V}{\partial t} \delta t + \left( \frac{\partial V}{\partial x}\right) \dot{x} \delta t.
$$

将展开式代入，并用 $\dot{x} = f(t, x, u)$ 替换，整理后两侧同时除以 $\delta t$，并令 $\delta t \to 0$，得到 **HJB 方程**

$$
- \frac{\partial V}{\partial t} (t, x) = \min_{u \in \mathcal{U}} \left[ L(t, x, u) + \left( \frac{\partial V}{\partial x}(t, x) \right)^{\top} f(t,x,u) \right].
$$

**与哈密顿量的关系**：如果我们将 $\frac{\partial V}{\partial x}$ 视作一个整体，则 HJB 方程可以更紧凑地写为：

$$
- \frac{\partial V}{\partial t} = \min_{u \in \mathcal{U}} H \left(t, x, u, \frac{\partial V}{\partial x}\right),
$$

变为一个 PDE，边界条件为 $V(t_f, x)  = \phi(t_f, x)$。

**求解与应用**：如果能求解上面的 PDE 得到 $V(t,x)$，则最优控制可以通过最小化哈密顿量在每个时刻 $(t,x)$ 得到

$$
u^\ast(t, x) = \arg\min_{u \in \mathcal{U}} H \left( t, x, u, \frac{\partial V}{\partial x}(t,x) \right).
$$

但是在高位 state space 中，HJB 方程的求解非常困难，这也被称为维数灾难 curse of dimensionality。
