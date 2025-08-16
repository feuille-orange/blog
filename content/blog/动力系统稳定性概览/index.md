+++

title = "动力系统稳定性概览"

date = "2025-08-16"

[taxonomies]

tags = ["Math", "ODE"]

+++

## 动力系统与平衡点

**动力系统**：在动力学系统理论中，我们一般考虑一个连续时间的动力系统

$$
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = {f}(\mathbf{x},t).
$$

其中 $\mathbf{x}(t)$ 是系统的状态 state，$t$ 是时间，${f}: D \to \mathbb{R}^n$ 是连续可微的向量场。

**自治动力系统**：通常更常见的动力系统是*自治 autonomous* 的，表示其行为只受状态 $\mathbf{x}(t)$ 的直接影响，而不受时间 $t$ 的直接影响，其形式为

$$
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = {f}(\mathbf{x}), \quad f(\mathbf{0}) = \mathbf{0}.
$$

不失一般性，我们这里直接令系统的初值为 $\mathbf{0}$。

**平衡点 Equilibrium Point**：若系统一旦到达某点 $\mathbf{x}_e \in D$，它将永远保持在该点，则该点被称为*平衡点 equilibrium point*，满足

$$
f(\mathbf{x}_e) = \mathbf{0}
$$

## 稳定性的概念

**稳定性的思想**：稳定性并不是一个单一的概念，而是有一个层次结构递进的。其核心思想是，当系统状态受到一个小的扰动 (perturbation) 偏离平衡点后，系统将如何响应。

- Lyapunov 稳定性：最基本的稳定性，直观含义为如果系统初始状态足够接近平衡点，那么它未来的所有状态都将保持在平衡点附近的一个任意小的邻域内
- Asymptotic Stability 渐进稳定：比 Lyapunov 稳定性更强，如果系统初始状态足够接近平衡点，系统不仅保持在平衡点附近，而且随着时间推移，最终会收敛到平衡点。
- Exponential Stability 指数稳定：比渐近稳定更强的稳定性，如果系统初始状态足够接近平衡点，系统不仅收敛到平衡点，而且其收敛速度至少像指数函数 $e^{-\lambda t}$ 一样快，其中 $\lambda > 0$。
- Global Stability 全局稳定：前述稳定性都是局部的，全局稳定性将这一范围扩展至整个状态空间 $\mathbb{R}^n$。

**Lyapunov 稳定性**：系统在平衡点 $\mathbf{x} = \mathbf{0}$ 是 Lyapunov 稳定的如果对于任意 $\epsilon > 0$，存在 $\delta > 0$，若 $\|\mathbf{x}(t_0)\| < \delta$ 满足

$$
\|\mathbf{x}(t) \| < \epsilon, \quad t \geq t_0.
$$

**Asymptotic Stability**：系统在平衡点 $\mathbf{x} = \mathbf{0}$ 是 Asymptotic 稳定的如果其是 Lyapunov 稳定的，且具有吸引性，即存在 $\delta > 0$，若 $\| \mathbf{x}(t_0) \| < \delta$，有

$$
\lim\limits_{t \to \infty} \mathbf{x}(t) = \mathbf{0}.
$$

**Exponential Stability**：系统在平衡点 $\mathbf{x} = \mathbf{0}$ 是指数稳定的如果存在 $\alpha > 0$，$\lambda > 0$，$\delta > 0$ 使得 $\|\mathbf{x}(t_0)\| < \delta$ 时，

$$
\|\mathbf{x}(t)\| \leq \alpha \|\mathbf{x}(t_0)\|e^{-\lambda (t - t_0)}.
$$

这里 $\lambda$ 称为 rate of convergence，$\alpha$ 反映系统的瞬态响应特性。

> 对于线性时不变系统 $\dot{\mathbf{x}} = A \mathbf{x}$​，**渐近稳定**等价于**指数稳定**，且当且仅当矩阵 A 的所有特征值都具有负实部。

**Global Stability**：系统在平衡点 $\mathbf{x} = \mathbf{0}$ 是*全局渐进稳定*​的如果其是 Lyapunov 稳定的，且对任意初始状态  $\mathbf{x}(t_0) \in \mathbb{R}^n$，有

$$
\lim\limits_{t \to \infty} \mathbf{x}(t) = \mathbf{0}
$$

类似的，也可以定义全局指数稳定。

## Lyapunov's Second Method

李雅普诺夫第二方法（也称直接法）允许我们不求解微分方程本身来分析稳定性。该方法的核心是寻找一个标量函数，即李雅普诺夫函数。

**径向无界 Radially Unbounded**：如果当 $\|\mathbf{x}\| \to \infty$ 时，$V(\mathbf{x}) \to \infty$，则称 $V(\mathbf{x})$ 是 radially unbounded 的。

**Lyapunov 稳定性定理**：考虑自治系统

$$
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = {f}(\mathbf{x}), \quad f(\mathbf{0}) = \mathbf{0}.
$$

设 $V(\mathbf{x})$ 是在包含原点的区域 $D \subseteq \mathbb{R}^n$ 上连续可微的标量函数。定义 $V(\mathbf{x})$ 沿系统轨迹的导数为

$$
\dot{V} = \frac{\mathrm{d}}{\mathrm{d} t}V(\mathbf{x}(t)) = \nabla V(\mathbf{x}) \cdot f(\mathbf{x}) = \frac{\partial V}{\partial \mathbf{x}} \dot{\mathbf{x}}.
$$

- 如果 $D$ 内 $V(\mathbf{x})$ 是正定，$\dot{V}(\mathbf{x})$ 是半负定的，则 $\mathbf{x} = \mathbf{0}$ 是 Lyapunov 稳定的；
- 如果 $D$ 内 $V(\mathbf{x})$ 是正定，$\dot{V}(\mathbf{x})$ 是负定的，则 $\mathbf{x} = \mathbf{0}$ 是渐进稳定的；
- 如果 $\mathbb{R}^n$ 内 $V(\mathbf{x})$ 是正定且径向无界，$\dot{V}(\mathbf{x})$ 是负定的，则 $\mathbf{x} = \mathbf{0}$ 是全局渐进稳定的；

‍
