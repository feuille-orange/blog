+++

title = "论文阅读：Method of Successive Approximations"

date = "2025-08-22"

[taxonomies]

tags = ["Optimal Control", "Machine Learning"]

+++



> Original Paper: [[1803.01299] An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks](https://arxiv.org/abs/1803.01299)

---

## The Optimal Control Viewpoint

**Neural Networks as Dynamical System**: Let $T \in \mathbb{Z}_+$ denote the number layers and $\{x_{s,0} \in \mathbb{R}^{d_0}: s = 0, 1, \cdots, S\}$ represent $S + 1$ inputs (images, time-series). Consider the dynamical system

$$
x_{s,t+1}=f_t(x_{s,t},\theta_t),\quad t=0,1,\ldots,T-1,
$$

where $f_{t}:\mathbb{R}^{d_{t}}\times\Theta_{t}\to\mathbb{R}^{d_{t+1}}$ is a transformation on the state, $\Theta_t$ is the trainable parameter set.

**Objective of Training**: The goal of training is to adjust the weights $\bm{\theta} := \{\theta_t: t = 0, 1, \cdots, T-1\}$ to minimize some loss function between final output $x_{s, T}$ and true targets $y_s$ of $x_{s,0}$.

**Statement of Problem**: Define $\Phi_s: \mathbb{R}^{d_T} \to \mathbb{R}$ that measures the loss, and the average loss function is

$$
\frac{1}{S}\sum_s \Phi_s(x_{s,T})
$$

We also consider some regularization terms $L_t: \mathbb{R}^{d_t} \times \Theta_t \to \mathbb{R}$, thus the problem is

$$
\min_{\boldsymbol{\theta}\in\boldsymbol{\Theta}}J(\boldsymbol{\theta}):=\frac{1}{S}\sum_{s=1}^{S}\Phi_{s}(x_{s,T})+\frac{1}{S}\sum_{s=1}^{S}\sum_{t=0}^{T-1}L_{t}(x_{s,t},\theta_{t}),
$$

$$
x_{s,t+1}=f_t(x_{s,t},\theta_t), \quad t=0,\cdots,T-1, \quad s\in \{1,2,\cdots,S\}
$$

where $\bm{\Theta} := \{\Theta_0 \times \cdots \times \Theta_{T-1}\}$.

---

## The Pontryagin’s Maximum Principle

**Hamiltonian Function**: Let $\bm{\theta}^\ast = \{\theta_0, \cdots, \theta_{T-1}\} \in \bm{\Theta}$ be a solution of the problem. For each $t$, define the Hamiltonian function $H_{t}:\mathbb{R}^{d_{t}}\times\mathbb{R}^{d_{t+1}}\times\Theta_{t}\to\mathbb{R}$​

$$
H_t(x,p,\theta):=p\cdot f_t(x,\theta)-\frac{1}{S}L_t(x,\theta).
$$

where $p \in \mathbb{R}^{d_{t+1}}$ is the co-state vector.

**Discrete PMP, Informal Statement**: Let $f_t$ and $\Phi_s, s = 1,2,\cdots, S$ be sufficiently smooth in $x$. Assume for each $t$ and $x \in \mathbb{R}^{d_t}$, the sets $\{f_t(x,\theta): \theta \in \Theta_t\}$ and $\{L_t(x,\theta): \theta \in \Theta_t\}$ are convex. Then there exists $\boldsymbol{p}_{s}^{*}:=\{p_{s,t}^{*}:t=0,\ldots,{T}\},$ such that

$$
x_{s,t+1}^* = \nabla_p H_t(x_{s,t}^*, p_{s,t+1}^*, \theta_t^*), \quad x_{s,0}^* = x_{s,0}
$$

$$
p_{s,t}^* = \nabla_x H_t(x_{s,t}^*, p_{s,t+1}^*, \theta_t^*), \quad p_{s,T}^* = -\frac{1}{S} \nabla \Phi_s(x_{s,T}^*)
$$

$$
\sum_{s=1}^S H_t(x_{s,t}^*, p_{s,t+1}^*, \theta_t^*) \geq \sum_{s=1}^S H_t(x_{s,t}^*, p_{s,t+1}^*, \theta), \forall \theta \in \Theta_t
$$

for $t = 0, 1, \cdots, T-1$ and $s = 1,2,\cdots, S$.

---

## The Method of Successive Approximations (MSA)

**Statement of MSA Algorithm**: Start from an initial guess $\boldsymbol{\theta}^{0}=\{\theta_{t}^{0}\in\Theta_{t}:t=0\ldots,T-1\}$,

- State Equation: $x_{s, t}$ means the state of the $s$-th sample at the $t$-th layer, $f_t$ is the transformation function at the $t$-th layer, $\theta_t$ is the control at the $t$-th layer

$$
x_{s,t+1}^{\boldsymbol{\theta}^0}=f_t(x_{s,t}^{\boldsymbol{\theta}^0},\theta_t^0),\quad x_{s,0}^{\boldsymbol{\theta}^0}=x_{s,0},
$$

- Co-State Equation: $p_{s,t}$ means the co-state of the $s$-th sample at the $t$-th layer, $\Phi_s$ measures the loss of the $s$-th sample, $H_t$ is the Hamiltonian function

$$
p_{s,t}^{\boldsymbol{\theta}^0}=\nabla_xH_t(x_{s,t}^{\boldsymbol{\theta}^0},p_{s,t+1}^{\boldsymbol{\theta}^0},\theta_t^0),\quad p_{s,T}^{\boldsymbol{\theta}^0}=-\frac{1}{S}\nabla\Phi_s(x_{s,T}^{\boldsymbol{\theta}^0}),
$$

- Maximization of the Hamiltonian:

$$
\theta_t^1=\arg\max_{\theta\in\Theta_t}\sum_{s=1}^SH_t(x_{s,t}^{\boldsymbol{\theta}^0},p_{s,t+1}^{\boldsymbol{\theta}^0},\theta),
\quad t=0,\ldots,T-1.
$$

> **MSA Algorithm**:
>
> - **Initialize**: $\boldsymbol{\theta}^{0}=\{\theta_{t}^{0}\in\Theta_{t}:t=0\ldots,T-1\};$
> - **For** $k = 0$ to $K$ **do**
>
>   - $x_{s,t+1}^{\boldsymbol{\theta}^k}=f_t(x_{s,t}^{\boldsymbol{\theta}^k},\theta_t^k)$ and $x_{s,0}^{\boldsymbol{\theta}^k}=x_{s,0}$ for all $s$ and $t$;
>   - $p_{s,t}^{\boldsymbol{\theta}^k}=\nabla_xH_t(x_{s,t}^{\boldsymbol{\theta}^k},p_{s,t+1}^{\boldsymbol{\theta}^k},\theta_t^k),p_{s,T}^{\boldsymbol{\theta}^k}=-\frac{1}{S}\nabla\Phi_s(x_{s,T})$ for all $s$ and $t$;
>   - $\theta_t^{k+1}=\arg\max_{\theta\in\Theta_t}\sum_{s=1}^SH_t(x_{s,t}^{\boldsymbol{\theta}^k},p_{s,t+1}^{\boldsymbol{\theta}^k},\theta)$ for $t = 0, \cdots, T-1$;
>
> - **End for**
