+++

title = "LaSalle's Invariance Principle"

date = "2025-08-18"

[taxonomies]

tags = ["Math", "ODE"]

+++



LaSalle's Invariance Principle 是 Lyapunov's Second Method 的延伸和推广。

## LaSalle's Invariance Principle 的核心思想

**回顾 Lyapunov's Second Method**：Lyapunov's Second Method 告诉我们，如果能为一个 Dynamical System 找到一个能量函数（Lyapunov 函数） $V(x)$，并且这个能量函数沿着系统的轨迹是**严格递减**的（$\dot{V}(x) < 0$），那么系统会渐进稳定（Asymptotically Stable）于能量最低点（通常指原点）。

**LaSalle's Invariance Principle 的思想**：很多时候 $V(x)$ 不一定是严格递减的，例如 $\dot{V}(x) \leq 0$，这意味着系统的能量在某些地方可能暂时不变，LaSalle's Invariance Principle 就是为了解决这个问题。其核心思想为：

- **能量不增且有下界**：$V(x)$ 永不增加且有下界，所以它一定收敛到某个常数；
- **寻找能量不变的区域**：既然能量最终不再变化，那么 $x(t)$ 一定趋近哪些让能量不再变化的点，即 $E = \{x: \dot{V}(x) = 0\}$；
- **在不变区域中寻找归宿**：系统轨迹 $x(t)$ 虽然会趋于 $E$，但是其不能在 $E$ 中乱跑，其最终肯定趋于 $E$ 内部的一个*不变集* $M$。即一旦 $x(t)$ 进入了 $M$，其就再也出不来了；
- **结论**：即使 $\dot{V}(x)$ 只是半负定，我们也能断定系统轨迹 $x(t)$ 最终收敛到 $E$ 中的最大不变集 $M$。

## 预备知识：不变集

**正不变集 Positively Invariant Set**：对于集合 $\Omega \subset \mathbb{R}^n$ 和 Dynamical System $\dot{x} = f(x)$，若对于任何从集合内部出发的 $x(0) \in \Omega$，其后的整个轨迹 $x(t)$ 都停留在 $\Omega$ 内部，则称其是 *positively invariant* 的。

**最大不变集 Largest Invariant Set**：给定集合 $E$，则 $M \subset E$ 是 $E$ 的*最大不变集*是所有完全包含于 $E$ 的系统轨迹的并。即如果一条轨迹始终在 $E$ 内部，则其属于 $M$，一旦其出过 $E$，则其就不属于 $M$ 了。

## LaSalle's Invariance Principle 定理

**LaSalle's Invariance Principle**：考虑 $\dot{x} = f(x)$，其中 $x \in D \subset \mathbb{R}^n$，$f:D \to \mathbb{R}^n$ 满足局部 Lipschitz 条件。若存在紧集 $\Omega \subset D$ 关于该系统 positively invariant，以及连续可微的 $V: \Omega \to \mathbb{R}$ 满足

$$
\dot{V}(x) = \nabla V(x) \cdot f(x) \leq 0, \quad \forall x \in \Omega.
$$

那么对于任意初始条件 $x(0) \in \Omega$，系统轨迹 $x(t)$ 都会收敛到 $E = \{x \in \Omega: \dot{V}(x) = 0\}$ 的最大不变集 $M$。

> 上述定理要求我们预先找到一个紧集 $\Omega$，但是在实际问题中往往很难构造出 $\Omega$，因此我们可以考虑下面更通用的版本。

**LaSalle's Invariance Principle (Generalized)** ：考虑 $\dot{x} = f(x)$，其中 $x \in D \subset \mathbb{R}^n$，$f:D \to \mathbb{R}^n$ 满足局部 Lipschitz 条件。若存在连续可微的 $V: D \to \mathbb{R}$ 满足

$$
\dot{V}(x) = \nabla V(x) \cdot f(x) \leq 0, \quad \forall x \in \Omega.
$$

且对于初始条件 $x(0) \in D$，若其轨迹 $x(t)$ 是**有界的**且始终保持在 $D$ 内，则 $x(t)$ 收敛到 $E = \{x \in \Omega: \dot{V}(x) = 0\}$ 的最大不变集 $M$。

**如何保证轨迹有界？** 若 $V(x)$ 是 Radially Unbounded 的，且 $\dot{V}(x) \leq 0$，则 $x(t)$ 是有界的。

> **径向无界 Radially Unbounded**：如果当 $\|{x}\| \to \infty$ 时，$V({x}) \to \infty$，则称 $V({x})$ 是 radially unbounded 的。
>
> 证明：因为当 $\|x\| \to \infty$ 时，$V(x) \to \infty$，这与 $\dot{V}(x) \leq 0$ 相矛盾。

‍
