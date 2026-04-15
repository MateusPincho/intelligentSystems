# Understanding the Zhang et al. (2020) Paper: A Student's Guide

# On the Identification of Noise Covariances and Adaptive Kalman Filtering

**Reference:** Zhang, L., Sidoti, D., Bienkowski, A., Pattipati, K. R., Bar-Shalom, Y., & Kleinman, D. L. (2020). *On the Identification of Noise Covariances and Adaptive Kalman Filtering: A New Look at a 50 Year-Old Problem*. IEEE Access, Vol. 8, pp. 59362-59388.

---

## 1. What Is This Paper About?

Imagine you have a robot moving through a room, and you're trying to track its position using a GPS sensor. You know the robot's motion model (e.g., "it moves forward at roughly constant velocity") and you know how the GPS converts the robot's true position into a measurement. But here's the problem: **how noisy is the robot's motion?** And **how noisy is the GPS?**

These two "noise levels" are described by **covariance matrices** called $Q$ (process noise — how unpredictable the motion is) and $R$ (measurement noise — how inaccurate the sensor is). The Kalman filter needs both of these to work optimally, but in practice, **nobody tells you what $Q$ and $R$ are**. You have to figure them out from the data itself.

This paper proposes a new method to **estimate $Q$ and $R$ from measured data**, using a clever six-step iterative algorithm. The method is more accurate than previous approaches that have been around for 50 years.

---

## 2. Background: The Linear Dynamic System

### 2.1 What Is a State?

In estimation theory, the **state** $x(k)$ is the complete description of the system at time step $k$. For example, for a moving robot in 1D, the state might be:

$$x(k) = \begin{bmatrix} \text{position}(k) \\ \text{velocity}(k) \end{bmatrix}$$

The state is what we **want to know** but **cannot observe directly**. We can only see noisy measurements.

### 2.2 The System Equations

The paper considers a **discrete-time linear system**, meaning the system evolves in fixed time steps ($k = 0, 1, 2, \ldots$) and the relationships are linear (no squares, exponentials, etc.):

**Process equation** (how the state evolves):

$$x(k+1) = F\,x(k) + \Gamma\,v(k) \tag{1}$$

**Measurement equation** (what the sensor sees):

$$z(k) = H\,x(k) + w(k) \tag{2}$$

Let's break down each piece:

| Symbol | Name | What it means | Dimensions |
|--------|------|---------------|------------|
| $x(k)$ | State vector | The true (hidden) state of the system at time $k$ | $n_x \times 1$ |
| $z(k)$ | Measurement vector | What the sensor actually reports at time $k$ | $n_z \times 1$ |
| $F$ | State transition matrix | Encodes the physics: "how does the state at time $k$ turn into the state at time $k+1$?" | $n_x \times n_x$ |
| $H$ | Measurement matrix | Encodes the sensor model: "which parts of the state does the sensor see?" | $n_z \times n_x$ |
| $\Gamma$ | Noise input matrix | Controls how the process noise enters the state. Sometimes the noise doesn't affect all states equally | $n_x \times n_v$ |
| $v(k)$ | Process noise | Random disturbance in the system dynamics (e.g., wind pushing the robot). Drawn from $\mathcal{N}(0, Q)$ | $n_v \times 1$ |
| $w(k)$ | Measurement noise | Random error in the sensor reading (e.g., GPS jitter). Drawn from $\mathcal{N}(0, R)$ | $n_z \times 1$ |

**Critical assumptions:**
- $v(k)$ and $w(k)$ are **white noise**: each sample is independent of all others (no memory)
- $v(k)$ and $w(k)$ are **mutually independent**: the sensor noise has nothing to do with the process noise
- Both are **zero-mean Gaussian**: their probability distributions are bell curves centered at zero
- $Q$ is positive semi-definite (its eigenvalues are $\geq 0$) and $R$ is positive definite (its eigenvalues are $> 0$)

### 2.3 A Concrete Example

For a robot moving in 1D with sampling period $T = 0.1$ seconds, under a "nearly constant velocity" model:

$$F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0.1 \\ 0 & 1 \end{bmatrix}$$

This says: "next position = current position + velocity $\times$ time" and "next velocity = current velocity" (constant velocity assumption).

$$\Gamma = \begin{bmatrix} T^2/2 \\ T \end{bmatrix} = \begin{bmatrix} 0.005 \\ 0.1 \end{bmatrix}$$

This says: the noise $v(k)$ represents a random acceleration, which affects both position (via $\frac{1}{2}aT^2$) and velocity (via $aT$).

$$H = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

This says: the sensor only measures position, not velocity.

---

## 3. The Kalman Filter: A Quick Review

### 3.1 What Does the Kalman Filter Do?

The Kalman filter produces the **best possible estimate** of the state $x(k)$ given all the measurements $z(1), z(2), \ldots, z(k)$ collected so far. "Best" means it minimizes the mean squared error.

It works in a two-step cycle: **predict**, then **correct**.

### 3.2 Predict Step (Time Update)

Before seeing the new measurement, we predict where the state should be based on the physics model:

$$\hat{x}(k+1|k) = F\,\hat{x}(k|k) \tag{3}$$

The notation $\hat{x}(k+1|k)$ means "our estimate of the state at time $k+1$, given measurements up to time $k$". This is a **one-step-ahead prediction**.

We also predict how uncertain we are about this prediction:

$$P(k+1|k) = F\,P(k|k)\,F' + \Gamma\,Q\,\Gamma' \tag{7}$$

Here $P(k+1|k)$ is the **prediction error covariance matrix** — it tells us how much we expect our prediction to be wrong. Notice that:
- The first term $F\,P(k|k)\,F'$ propagates our current uncertainty through the dynamics
- The second term $\Gamma\,Q\,\Gamma'$ **adds** uncertainty from the process noise

This is why $Q$ matters so much: it controls how fast uncertainty grows between measurements.

### 3.3 Innovation

When the new measurement $z(k+1)$ arrives, we compute the **innovation** (also called the "residual" or "prediction error"):

$$\nu(k+1) = z(k+1) - H\,\hat{x}(k+1|k) \tag{4}$$

The innovation is the **surprise**: the difference between what we actually measured and what we predicted we would measure. If the filter is working well, the innovations should look like random noise with no patterns.

The **innovation covariance** tells us how large we expect the innovations to be:

$$S(k+1) = H\,P(k+1|k)\,H' + R \tag{7}$$

Notice that $S$ depends on both the prediction uncertainty (through $P$) and the measurement noise (through $R$).

### 3.4 Correct Step (Measurement Update)

Now we blend our prediction with the new measurement:

$$\hat{x}(k+1|k+1) = \hat{x}(k+1|k) + W(k+1)\,\nu(k+1) \tag{5}$$

The **Kalman gain** $W$ determines how much we trust the measurement vs. our prediction:

$$W(k+1) = P(k+1|k)\,H'\,S(k+1)^{-1} \tag{8}$$

Think of it this way:
- If the measurement is very precise ($R$ is small), $S$ is small, so $W$ is large — we trust the measurement more
- If the prediction is very uncertain ($P$ is large), $W$ is also large — we lean more on the measurement
- If the measurement is noisy ($R$ is large), $W$ is small — we trust our prediction more

The updated covariance is:

$$P(k+1|k+1) = P(k+1|k) - W(k+1)\,S(k+1)\,W(k+1)' \tag{9}$$

There is also a more numerically stable form called the **Joseph form**:

$$P(k+1|k+1) = (I - WH)\,P(k+1|k)\,(I - WH)' + W\,R\,W' \tag{13}$$

This is mathematically equivalent but avoids some numerical issues with floating-point arithmetic.

### 3.5 Why Does the Filter Need $Q$ and $R$?

Look at the equations above: $Q$ appears in the prediction covariance (eq. 7), and $R$ appears in the innovation covariance (eq. 7) and the Kalman gain (eq. 8, through $S$). **If $Q$ and $R$ are wrong, the Kalman gain $W$ will be wrong**, and the filter will produce suboptimal estimates.

- If $Q$ is **too large**: the filter thinks the dynamics are very unpredictable, so it trusts the measurements too much $\rightarrow$ noisy estimates
- If $Q$ is **too small**: the filter is overconfident in its predictions and ignores useful measurement information $\rightarrow$ sluggish, biased estimates
- Similar issues occur with incorrect $R$

---

## 4. Steady-State Kalman Filter

### 4.1 What Is "Steady State"?

In the standard Kalman filter, the gain $W(k)$ and covariance $P(k)$ change at every time step. But for a **time-invariant system** (where $F$, $H$, $\Gamma$, $Q$, $R$ don't change over time), something nice happens: after enough time steps, $W(k)$ and $P(k)$ **converge to constant values**. This is the "steady state".

The steady-state prediction covariance $\bar{P}$ satisfies the **Discrete Algebraic Riccati Equation** (DARE):

$$\bar{P} = F\left[\bar{P} - \bar{P}H'(H\bar{P}H' + R)^{-1}H\bar{P}\right]F' + \Gamma Q \Gamma' \tag{10}$$

This is a fixed-point equation: $\bar{P}$ appears on both sides. It can be solved numerically using `scipy.linalg.solve_discrete_are`.

Once we have $\bar{P}$, the steady-state quantities are:

$$S = H\bar{P}H' + R \tag{15}$$

$$W = \bar{P}H'S^{-1} \tag{14}$$

$$P = \bar{P} - WSW' \tag{12}$$

### 4.2 The Closed-Loop Matrix $\bar{F}$

An important matrix that appears throughout the paper is:

$$\bar{F} = F(I_{n_x} - WH) \tag{16}$$

This is the **closed-loop state transition matrix**. Think of it this way: without the filter, the state evolves as $F\,x$. With the filter correcting the estimate at each step, the **estimation error** evolves as $\bar{F}\,\tilde{x}$, where $\tilde{x}$ is the error.

For the filter to be **stable** (errors don't grow unboundedly), the eigenvalues of $\bar{F}$ must be **inside the unit circle** (i.e., their absolute values must be less than 1).

### 4.3 Why Use the Steady-State Filter?

The entire six-step algorithm in this paper operates on the **steady-state** Kalman filter. Instead of a time-varying gain $W(k)$, we use a single constant gain $W$. This simplification:
- Makes the innovation sequence **stationary** (its statistical properties don't change over time), which is needed for computing sample statistics
- Reduces the number of unknowns from a sequence of gains to a single matrix

---

## 5. Can We Actually Estimate $Q$ and $R$? (Identifiability)

### 5.1 The Key Question

Before trying to estimate $Q$ and $R$, we need to ask: **is it even theoretically possible?** Maybe different combinations of $Q$ and $R$ produce the same innovation sequence, making it impossible to tell them apart. This is the **identifiability** question.

### 5.2 The Minimal Polynomial

The paper's approach to identifiability is based on the **minimal polynomial** of the closed-loop matrix $\bar{F}$.

**What is a minimal polynomial?** Every square matrix satisfies its own characteristic polynomial (this is the Cayley-Hamilton theorem). But there might be a **lower-degree** polynomial that also annihilates the matrix. The minimal polynomial is the lowest-degree monic polynomial $p(\lambda)$ such that $p(\bar{F}) = 0$.

For a matrix $\bar{F}$ with distinct eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_m$, the minimal polynomial is simply:

$$p(\lambda) = (\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_m)$$

Written in the form used by the paper:

$$\sum_{i=0}^{m} a_i \bar{F}^{m-i} = 0, \quad a_0 = 1 \tag{21}$$

where $m$ is the degree and $a_1, \ldots, a_m$ are the coefficients.

**How to compute it:** Find the eigenvalues of $\bar{F}$, keep only the distinct ones (with tolerance $\sim 10^{-8}$ to handle numerical noise), and form the polynomial from its roots using `numpy.poly()`.

### 5.3 Why Does the Minimal Polynomial Matter?

The minimal polynomial lets us **transform the innovation sequence** into something simpler. By applying the minimal polynomial as a filter to the innovations:

$$\xi(k) = \sum_{i=0}^{m} a_i \nu(k-i) \tag{23}$$

we get a sequence $\xi(k)$ that is a **finite moving average** of the noise processes $v(k)$ and $w(k)$:

$$\xi(k) = \sum_{l=1}^{m} \mathscr{B}_l\, v(k-l) + \sum_{l=0}^{m} \mathscr{G}_l\, w(k-l) \tag{27}$$

The key insight is that the lag covariances of $\xi(k)$ depend **linearly** on $Q$ and $R$, which is what makes the estimation problem tractable.

The matrices $\mathscr{B}_l$ and $\mathscr{G}_l$ are computed from the known system matrices:

$$\mathscr{B}_l = H\left(\sum_{i=0}^{l-1} a_i \bar{F}^{l-i-1}\right)\Gamma \tag{28}$$

$$\mathscr{G}_l = a_l I_{n_z} - H\left(\sum_{i=0}^{l-1} a_i \bar{F}^{l-i-1}\right)FW \tag{29}$$

$$\mathscr{G}_0 = I_{n_z} \tag{30}$$

### 5.4 The Identifiability Matrix $\mathscr{I}$

The lag covariances $L_j = E[\xi(k)\xi(k-j)']$ can be written as a **linear function** of the unknown elements of $Q$ and $R$:

$$\mathscr{I} \begin{bmatrix} \text{vec}(Q) \\ \text{vec}(R) \end{bmatrix} = \begin{bmatrix} L_0 \\ L_1 \\ \vdots \\ L_m \end{bmatrix} \tag{36}$$

The matrix $\mathscr{I}$ is called the **noise covariance identifiability matrix**. It is constructed from products of columns of $\mathscr{B}_l$ and $\mathscr{G}_l$ (Algorithm 1 in the paper).

**The identifiability condition is:**

$$\text{rank}(\mathscr{I}) - n_R \geq n_Q \tag{37}$$

where $n_Q$ is the number of unknowns in $Q$ and $n_R$ is the number of unknowns in $R$. If this condition holds, then $Q$ and $R$ can be uniquely determined from the data.

**Practical note:** The rank of $\mathscr{I}$ does not depend on $W$, so for simplicity you can compute $\mathscr{I}$ with $W = 0$.

**Example:** For Case 2 (Neethling), $\mathscr{I}$ is a $3 \times 2$ matrix with rank 2, and there are 2 unknowns ($Q$ is scalar, $R$ is scalar), so identifiability holds easily.

---

## 6. The Heart of the Algorithm: Estimating $W$

### 6.1 The Core Idea

A properly tuned Kalman filter produces innovations $\nu(k)$ that are **white noise** — meaning they are uncorrelated at all time lags. If the innovations are correlated, the filter is suboptimal. So we can judge how good a gain $W$ is by measuring **how white the innovations are**.

### 6.2 Sample Autocovariances

Given an innovation sequence $\{\nu(k)\}_{k=1}^{N}$, we compute the **sample autocovariance** at lag $i$:

$$\hat{C}(i) = \frac{1}{N-M} \sum_{j=1}^{N-M} \nu(j)\,\nu(j+i)' \quad \text{for } i = 0, 1, \ldots, M-1 \tag{50}$$

- $\hat{C}(0)$ is the sample variance of the innovations (this approximates $S$)
- $\hat{C}(i)$ for $i \geq 1$ measures the correlation between innovations that are $i$ steps apart
- For an optimal filter, $\hat{C}(i) \approx 0$ for $i \geq 1$ (innovations are white)
- $M$ is the number of lags to compute — it should be $\geq n_x$ (number of states), typically 40-100

### 6.3 The Objective Function $J$

We want to minimize a measure of "how non-white" the innovations are. The paper defines:

$$J = \frac{1}{2} \text{tr}\left\{ \sum_{i=1}^{M-1} \left[\text{diag}(\hat{C}(0))\right]^{-1/2} \hat{C}(i)'\, \left[\text{diag}(\hat{C}(0))\right]^{-1}\, \hat{C}(i)\, \left[\text{diag}(\hat{C}(0))\right]^{-1/2} \right\} \tag{52}$$

This looks intimidating, but let's unpack it:

1. **$\hat{C}(i)$ for $i \geq 1$**: these are the off-zero-lag autocovariances. If the innovations were perfectly white, all of these would be zero, and $J = 0$.
2. **$\text{diag}(\hat{C}(0))$**: this is the diagonal of the zero-lag covariance, used for **normalization**. Without normalization, a measurement channel with large variance would dominate $J$.
3. **The trace of the matrix product**: this is a scalar that sums up the squared, normalized correlations across all lags and all measurement channels.

**Bottom line:** $J$ measures the total amount of correlation in the innovation sequence. **$J = 0$ means perfect whiteness** (optimal filter). We want to find $W$ that minimizes $J$.

To make the gradient computation tractable, define the following helper quantities:

$$\mathscr{E} = \left[\text{diag}(\hat{C}(0))\right]^{-1/2} \tag{59}$$

This is a diagonal matrix whose entries are $1/\sqrt{\hat{C}(0)_{ii}}$.

$$\Phi(i) = H\bar{F}^{i-1}F \tag{56}$$

$$\Theta(i) = \Phi(i)'\,\mathscr{E}^2\,\Phi(i) \tag{55}$$

$$\Psi = X - W\hat{C}(0) \tag{57}$$

where $X$ is a matrix obtained from least-squares (explained next).

### 6.4 The Correlation Residual $X$

The theoretical autocovariance at lag $i$ has the structure $C(i) = H\bar{F}^{i-1}F \cdot [PH' - WC(0)]$. This means $X = PH'$ can be estimated by solving the **least-squares problem**:

$$\underbrace{\begin{bmatrix} HF \\ H\bar{F}F \\ H\bar{F}^2F \\ \vdots \\ H\bar{F}^{M-2}F \end{bmatrix}}_{\text{call this } A} X = \underbrace{\begin{bmatrix} \hat{C}(1) \\ \hat{C}(2) \\ \hat{C}(3) \\ \vdots \\ \hat{C}(M-1) \end{bmatrix}}_{\text{call this } B} \tag{62}$$

The solution is:

$$X = A^{\dagger} B = (A'A)^{-1}A'B \tag{63}$$

where $A^{\dagger}$ is the Moore-Penrose pseudoinverse of $A$. In Python, use `numpy.linalg.pinv(A) @ B`.

**Intuition:** We're fitting the theoretical lag structure to the observed sample autocovariances. The matrix $X$ captures the "residual correlation" — what the current gain $W$ fails to explain.

### 6.5 The Gradient $\nabla_W J$

To minimize $J$, we need its gradient with respect to $W$. The derivation (Appendix E of the paper) is lengthy, but the result is:

$$\nabla_W J = -\left[\sum_{i=1}^{M-1}\Theta(i)\, X\, \mathscr{E}^2\, \hat{C}(0) + F'\, Z\, F\, X\right] \tag{60}$$

where $Z$ is the solution to a **discrete Lyapunov equation**:

$$Z = \bar{F}'\,Z\,\bar{F} + \text{(a known matrix that depends on current } W, \hat{C}, \Theta, \mathscr{E}\text{)} \tag{61}$$

The right-hand side of the Lyapunov equation is:

$$\frac{1}{2}\sum_{i=1}^{M-1}\left[\Theta(i)(\Psi - W\hat{C}(0))\mathscr{E}^2 H + H'\mathscr{E}^2(\Psi' - \hat{C}(0)W')\Theta(i)\right]$$

**What is a Lyapunov equation?** It's an equation of the form $Z = A'ZA + Q_{rhs}$, where $A$ and $Q_{rhs}$ are known and $Z$ is unknown. It has a unique solution when $A$ is stable (eigenvalues inside the unit circle). In Python, use `scipy.linalg.solve_discrete_lyapunov`.

### 6.6 Gradient Descent with Bold Driver Step Size

Once we have the gradient, we update $W$ using **gradient descent**:

$$W^{(r+1)} = W^{(r)} - \alpha^{(r)}\,\nabla_W J \tag{135}$$

The step size $\alpha^{(r)}$ uses the **bold driver** method, which is an adaptive scheme:

**Initialization:**

$$\alpha^{(0)} = \min\left(c\left(\frac{N}{N_s}\right)^\beta,\; c_{\max}\right) \tag{136}$$

With default values $c = 0.01$, $N_s = N$, $\beta = 2$, $c_{\max} = 0.2$, this gives $\alpha^{(0)} = 0.01$.

**Adaptation at each iteration:**

$$\alpha^{(r)} = \begin{cases} 0.5 \cdot \alpha^{(r-1)} & \text{if } J^{(r)} > J^{(r-1)} \quad \text{(we overshot — step was too big)} \\ \min(1.1 \cdot \alpha^{(r-1)},\; \bar{c}) & \text{otherwise} \quad \text{(we're doing well — try a slightly larger step)} \end{cases} \tag{137}$$

**Intuition:** If the last step made things worse ($J$ increased), we halve the step size to be more cautious. If things improved, we increase the step size by 10% to move faster (but never exceed $\bar{c}$).

### 6.7 Termination Conditions

The gradient descent stops when **any** of these five conditions is met:

1. **Gain converged:** The relative change in $W$ is tiny: $\delta_W < \zeta_W = 10^{-6}$
2. **Gradient vanished:** $\|\nabla_W J\| < \zeta_\Delta = 10^{-6}$ (we're at a stationary point)
3. **Objective small enough:** $J < \zeta_J = 10^{-6}$ (innovations are already very white)
4. **Patience exhausted:** $J$ hasn't improved for 5 consecutive iterations (we're stuck)
5. **Max iterations reached:** We've done $n_L$ iterations (typically 100-500)

### 6.8 Best Gain Selection

Throughout the iterations, we keep track of the **best** $(W^*, J^*)$ — the gain that achieved the lowest objective value. When the algorithm terminates, we use $W^*$, not necessarily the last iterate:

$$W = \arg\min_r J^{(r)} \tag{139}$$

This is important because the bold driver method might occasionally overshoot.

---

## 7. Estimating $R$: Five Ways to Skin a Cat

### 7.1 Post-Fit Residuals

Once we have the optimal gain $W$, we can compute not just the innovations $\nu(k)$ but also the **post-fit residuals**:

$$\mu(k) = z(k) - H\hat{x}(k|k) \tag{65}$$

This is the difference between the measurement and our **updated** estimate (after incorporating the measurement). Compare this to the innovation $\nu(k) = z(k) - H\hat{x}(k|k-1)$, which uses the **predicted** estimate (before incorporating the measurement).

The relationship between them is:

$$\mu(k) = (I_{n_z} - HW)\,\nu(k) \tag{66}$$

### 7.2 Why Post-Fit Residuals Are Useful

The paper proves that the covariance of the post-fit residual $G = E[\mu(k)\mu(k)']$ satisfies a beautiful relationship:

$$G = R - HPH' \tag{74}$$

and more importantly:

$$G = R\,S^{-1}\,R \tag{85}$$

This means we can estimate $R$ from $G$ and $S$, both of which we can compute from samples!

### 7.3 The Five Formulas

The paper derives five theoretically equivalent but numerically different formulas for $R$:

**R1** — Directly from the gain and innovation covariance:

$$R = (I_{n_z} - HW)\,S \tag{76}$$

**R2** — From the cross-covariance of innovations and post-fit residuals:

$$R = \frac{1}{2}\left[E[\mu(k)\nu(k)'] + E[\nu(k)\mu(k)']\right] \tag{77}$$

**R3** — From the "matrix geometric mean" equation:

$$G = R\,S^{-1}\,R \tag{78}$$

**R4** — An average of direct estimates:

$$R = \frac{1}{2}\left[G + S - HWSW'H'\right] \tag{79}$$

**R5** — A symmetrized version:

$$R = \frac{1}{2}\left[G(I_{n_z} - W'H')^{-1} + (I_{n_z} - HW)^{-1}G\right] \tag{80}$$

### 7.4 Why R3 Is Recommended

**R3** is the recommended method because it **guarantees positive definiteness** of the estimated $R$. The other methods can produce estimates with negative eigenvalues due to numerical errors, which is physically nonsensical (a covariance matrix must be positive definite).

The equation $G = RS^{-1}R$ is equivalent to a **continuous-time algebraic Riccati equation** (CARE), which always has a positive definite solution when $G$ and $S$ are positive definite.

### 7.5 How to Solve R3 (Cholesky + Eigendecomposition)

This is the recommended numerical procedure (Appendix F):

**Step 1:** Compute the Cholesky decomposition of $S^{-1}$:

$$S^{-1} = \mathscr{L}\mathscr{L}'$$

This factors the symmetric positive definite matrix $S^{-1}$ into a lower-triangular matrix $\mathscr{L}$ times its transpose. Think of it as a "matrix square root".

**Step 2:** Note that the equation $G = RS^{-1}R$ can be rewritten as:

$$\mathscr{L}' G \mathscr{L} = (\mathscr{L}' R \mathscr{L})^2$$

So we need the "matrix square root" of $\mathscr{L}' G \mathscr{L}$.

**Step 3:** Eigendecompose $\mathscr{L}' G \mathscr{L}$:

$$\mathscr{L}' G \mathscr{L} = U \Lambda U'$$

where $\Lambda$ is diagonal with non-negative eigenvalues and $U$ is orthogonal.

**Step 4:** Take the square root:

$$\mathscr{L}' R \mathscr{L} = U \Lambda^{1/2} U'$$

**Step 5:** Recover $R$:

$$R = (\mathscr{L}')^{-1}\, U\, \Lambda^{1/2}\, U'\, \mathscr{L}^{-1}$$

**Python implementation sketch:**
```python
L = cholesky(inv(S), lower=True)      # Step 1
M_mat = L.T @ G_hat @ L               # Step 2
eigvals, U = eigh(M_mat)              # Step 3
sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))  # Step 4
R_est = solve_triangular(L.T, U @ np.diag(sqrt_eigvals) @ U.T, lower=False)
R_est = solve_triangular(L, R_est, lower=True, trans='T')  # Step 5
```

### 7.6 Diagonal $R$

If $R$ is known to be diagonal (which is common — sensor noise channels are often independent), simply keep only the diagonal elements of the full $R$ estimate.

---

## 8. Estimating $Q$ and $P$: An Iterative Dance

### 8.1 Why Is This Hard?

Unlike $R$, the process noise covariance $Q$ cannot be estimated in closed form in the general case. The reason is that $Q$ and the state covariance $P$ are **coupled** through the Riccati equation: you need $Q$ to compute $P$, but you also need $P$ to compute $Q$. This chicken-and-egg problem is resolved by **iterating**.

### 8.2 Initialization

We start with a reasonable initial guess for $Q$ using the **Wiener process approximation**:

$$\Gamma\,Q^{(0)}\,\Gamma' = W\,S\,W' \tag{165}$$

To extract $Q^{(0)}$ from $\Gamma\,Q^{(0)}\,\Gamma'$, we use the pseudoinverse of $\Gamma$:

$$Q^{(0)} = \Gamma^{\dagger}\,(W\,S\,W')\,(\Gamma')^{\dagger}$$

where $\Gamma^{\dagger} = (\Gamma'\Gamma)^{-1}\Gamma'$ is the Moore-Penrose pseudoinverse (use `numpy.linalg.pinv`).

Then we initialize $P^{(0)}$ by solving the **discrete Lyapunov equation**:

$$P^{(0)} = \bar{F}\,P^{(0)}\,\bar{F}' + WRW' + (I_{n_x} - WH)\,\Gamma\,Q^{(0)}\,\Gamma'\,(I_{n_x} - WH)' \tag{122}$$

This equation has the form $P = APA' + B$ with $A = \bar{F}$ and $B$ known. Solve it with `scipy.linalg.solve_discrete_lyapunov`.

### 8.3 The Inner Loop: Iterating $P$

For a fixed $Q^{(t)}$, we iterate to find the consistent $P$:

$$P^{(\ell+1)} = \left[\left(F\,P^{(\ell)}\,F' + \Gamma\,Q^{(t)}\,\Gamma'\right)^{-1} + H'\,R^{-1}\,H\right]^{-1} \tag{123}$$

**What is this doing?** At each step, we:
1. Compute the prediction covariance $\bar{P}^{(\ell)} = F\,P^{(\ell)}\,F' + \Gamma\,Q^{(t)}\,\Gamma'$ (propagate through dynamics)
2. Invert it, add $H'R^{-1}H$ (incorporate measurement information)
3. Invert back to get $P^{(\ell+1)}$

This is really just applying the steady-state update formula repeatedly. Iterate until $\|P^{(\ell+1)} - P^{(\ell)}\| < \text{tol}$ (convergence).

### 8.4 Updating $Q$

Once $P$ has converged, update $Q$ using the relationship derived from $\bar{P} = FPF' + \Gamma Q \Gamma'$ and $P = \bar{P} - WSW'$:

$$D^{(t+1)} = P + W\,S\,W' - F\,P\,F' \tag{124}$$

$$Q^{(t+1)} = \Gamma^{\dagger}\,D^{(t+1)}\,(\Gamma')^{\dagger} \tag{125}$$

**Intuition:** $D = \Gamma Q \Gamma'$ is the "innovation added by process noise" in state space. We recover $Q$ by "undoing" the $\Gamma$ transformation.

### 8.5 Structural Constraints: The Mask Matrix

Often we know something about the **structure** of $Q$. For example, if $Q$ is diagonal (the process noise components are independent), we can enforce this:

$$Q^{(t+1)} = A \odot Q^{(t+1)} \tag{126}$$

where $A$ is a **mask matrix** — a binary matrix with 1s where the element should be kept and 0s where it should be forced to zero. For a diagonal constraint, $A = I$ (the identity matrix). The symbol $\odot$ denotes the **Hadamard product** (element-wise multiplication).

### 8.6 Regularization for Ill-Conditioned Systems

Some systems have identifiability matrices $\mathscr{I}$ with high condition numbers, meaning the estimation is very sensitive to noise. For these cases, add a regularization term:

$$Q^{(t+1)} = \Gamma^{\dagger}\,\left[D^{(t+1)} + \lambda_Q\,I_{n_x}\right]\,(\Gamma')^{\dagger} \tag{127}$$

The parameter $\lambda_Q$ (typically 0.1-0.3) acts like a "safety net" that prevents $Q$ from becoming too small or negative. Then apply the mask as usual.

### 8.7 The Full Q/P Iteration

1. Initialize $Q^{(0)}$ from eq. (165) and $P^{(0)}$ from eq. (122)
2. **Repeat** (index $t = 0, 1, 2, \ldots$):
   a. **Inner loop**: fix $Q^{(t)}$, iterate $P^{(\ell)}$ via eq. (123) until convergence
   b. **Update $Q$**: compute $D^{(t+1)}$ from eq. (124), then $Q^{(t+1)}$ from eq. (125)
   c. Apply mask: $Q^{(t+1)} = A \odot Q^{(t+1)}$
   d. Check convergence: $\|Q^{(t+1)} - Q^{(t)}\| < \text{tol}$?
3. Once converged, compute $\bar{P} = FPF' + \Gamma Q \Gamma'$

---

## 9. Putting It All Together: The Six-Step Algorithm

Now we combine everything into the complete algorithm.

### Initialization

1. Choose initial guesses $Q^{(0)}_{\text{init}}$ and $R^{(0)}_{\text{init}}$ (they don't need to be accurate)
2. Solve DARE with these guesses to get the initial gain $W^{(0)}$
3. This ensures $\bar{F} = F(I - W^{(0)}H)$ is stable from the start

### Step 1: Run the Kalman Filter

With the current constant gain $W^{(r)}$, process all $N$ measurements:

$$\hat{x}^{(r)}(k+1|k) = F\,\hat{x}^{(r)}(k|k) \tag{128}$$

$$\nu^{(r)}(k+1) = z(k+1) - H\,\hat{x}^{(r)}(k+1|k) \tag{129}$$

$$\hat{x}^{(r)}(k+1|k+1) = \hat{x}^{(r)}(k+1|k) + W^{(r)}\,\nu^{(r)}(k+1) \tag{130}$$

$$\mu^{(r)}(k+1) = z(k+1) - H\,\hat{x}^{(r)}(k+1|k+1) \tag{131}$$

**Note:** The gain $W^{(r)}$ is **constant** throughout the entire pass — we don't update it at each time step. This is the key difference from a standard Kalman filter.

**Burn-in:** The first 20-50 samples should be discarded when computing statistics, because the filter needs time to "forget" the initial state estimate.

### Step 2: Compute Sample Autocovariances

From the innovation sequence $\{\nu^{(r)}(k)\}$:

$$\hat{C}(i) = \frac{1}{N-M}\sum_{j=1}^{N-M} \nu(j)\,\nu(j+i)' \quad \text{for } i = 0, 1, \ldots, M-1 \tag{50}$$

### Step 3: Update $W$ via Gradient Descent

1. Compute the objective $J$ from $\hat{C}(i)$
2. Compute the helper matrices $\Phi(i)$, $\Theta(i)$, $\mathscr{E}$, $X$, $\Psi$
3. Solve the Lyapunov equation for $Z$
4. Compute the gradient $\nabla_W J$
5. Update $W$ using the bold driver step size
6. Repeat until a termination condition is met
7. Select the best $W$ (the one with the lowest $J$)

### Step 4: Estimate $R$

1. With the converged $W$, set $S = \hat{C}(0)$
2. Compute the sample post-fit residual covariance $\hat{G} = \frac{1}{N}\sum_{k=1}^{N}\mu(k)\mu(k)'$
3. Solve $\hat{G} = R\,S^{-1}\,R$ for $R$ using the Cholesky + eigendecomposition method
4. If $R$ is known to be diagonal, keep only diagonal elements

### Step 5: Estimate $Q$ and $P$

1. Initialize $Q^{(0)}$ using the Wiener process formula
2. Solve for $P^{(0)}$ via Lyapunov equation
3. Alternate between iterating $P$ and updating $Q$ until convergence
4. Compute $\bar{P} = FPF' + \Gamma Q \Gamma'$

### Step 6: Outer Successive Approximation

1. With the new $(Q, R)$ estimates, solve DARE to get a new initial $W^{(0)}$
2. Go back to Step 1 and repeat
3. Track the best $J$ across all outer iterations
4. Stop when:
   - The improvement in $J$ between outer iterations is below $\zeta_J = 10^{-6}$
   - OR 20 outer iterations have been completed
5. Return the estimates $(W, S, R, Q, \bar{P}, P)$ corresponding to the best $J$

---

## 10. Special Cases with Closed-Form Solutions

### 10.1 Wiener Process ($F = I$, $H = I$)

When both the state transition and measurement matrices are identity matrices, everything simplifies dramatically and **no iteration is needed**.

The system becomes:
$$x(k+1) = x(k) + v(k) \quad \text{(random walk)}$$
$$z(k) = x(k) + w(k) \quad \text{(direct observation)}$$

Define the first difference of measurements:

$$\xi(k) = z(k) - z(k-1) = v(k) + (W - I)\nu(k-1) \tag{143-145}$$

Compute the sample lag-0 and lag-1 covariances of $\xi(k)$: $L_0$ and $L_1$.

Then all quantities can be computed in closed form:

1. Solve for $S$ from: $L_0 = S + L_1 S^{-1} L_1'$ (this is a Riccati equation with positive definite solution)
2. $W = I + L_1 S^{-1}$
3. $\bar{P} = WS$
4. $Q = WSW'$
5. $R$ from R3: $G = RS^{-1}R$, where $G = (I - W)S(I - W)'$

### 10.2 General $F$, $H = I$ Case

When only $H = I$ (we observe all states directly, but dynamics are non-trivial):

Define $\xi(k) = z(k) - Fz(k-1)$, then:

$$L_0 = \Gamma Q\Gamma' + R + FRF' \tag{185}$$
$$L_1 = -\bar{F}S \tag{180}$$

And similarly solve for $S$, then $W$, then $R$ and $Q$.

---

## 11. How to Verify the Results

### 11.1 Filter Stability Check

At every iteration, verify that the eigenvalues of $\bar{F} = F(I - WH)$ are **inside the unit circle**:

```python
eigvals = np.linalg.eigvals(F @ (np.eye(n_x) - W @ H))
assert np.all(np.abs(eigvals) < 1.0), "Filter is unstable!"
```

### 11.2 Positive Definiteness Check

The estimated $Q$ and $R$ must have all positive eigenvalues:

```python
assert np.all(np.linalg.eigvalsh(R_est) > 0), "R is not positive definite!"
assert np.all(np.linalg.eigvalsh(Q_est) > 0), "Q is not positive definite!"
```

### 11.3 Normalized Innovation Squared (NIS)

The **NIS** at time $k$ is:

$$\varepsilon(k) = \nu(k)'\,S^{-1}\,\nu(k) \tag{191}$$

For a consistent (well-tuned) filter, $\varepsilon(k)$ should follow a $\chi^2$ distribution with $n_z$ degrees of freedom, so its expected value is $n_z$.

The **averaged NIS** over Monte Carlo runs:

$$\bar{\varepsilon}(k) = \frac{1}{n_{MC}}\sum_{i=1}^{n_{MC}} \nu_i(k)'\,S_i^{-1}\,\nu_i(k)$$

Plot $\bar{\varepsilon}(k)/n_z$ over time — it should hover around 1 and stay within the 95% confidence band.

### 11.4 Identifiability Check

Before running the algorithm, verify:

$$\text{rank}(\mathscr{I}) - n_R \geq n_Q$$

If this fails, the problem is ill-posed and the algorithm cannot be expected to find unique $Q$ and $R$.

### 11.5 Truth Within Confidence Interval

For Monte Carlo studies, the **true values** of $Q$, $R$, $W$, $\bar{P}$ should lie within the 95% probability interval computed from the distribution of estimates across runs.

---

## 12. Hyperparameters Reference Table

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| $M$ | 40-100 ($\geq n_x$) | Number of autocovariance lags. More lags = more information but noisier estimates |
| $N$ | 1000-10000 | Number of data samples. More = better estimates |
| $n_L$ | 100-500 | Max gradient descent iterations for $W$. Scale with $n_x$ |
| $\zeta_J$ | $10^{-6}$ | Stop if objective $J$ drops below this |
| $\zeta_W$ | $10^{-6}$ | Stop if gain change is below this |
| $\zeta_\Delta$ | $10^{-6}$ | Stop if gradient norm is below this |
| patience | 5 | Stop if no $J$ improvement for this many iterations |
| $c$ | 0.01 | Base initial step size |
| $c_{\max}$ | 0.2 | Maximum allowed step size |
| $\beta$ | 2 | Step-size exponent |
| $N_s$ | $N$ | Step-size normalization factor |
| $\lambda_Q$ | 0-0.3 | $Q$ regularization (0 for well-conditioned, 0.1-0.3 for ill-conditioned) |
| Outer loop limit | 20 | Max outer successive approximation iterations |

---

## 13. Common Pitfalls and Tips

### 13.1 DARE Convention in SciPy

`scipy.linalg.solve_discrete_are` uses a **transposed convention** compared to most control textbooks. To solve for the prediction covariance $\bar{P}$:

```python
P_bar = solve_discrete_are(F.T, H.T, Gamma @ Q @ Gamma.T, R)
```

Note the transposes on $F$ and $H$. Always verify on a simple scalar example before trusting the results.

### 13.2 Initial Gain Must Stabilize the Filter

**Never** initialize $W$ randomly. Always bootstrap from DARE with some reasonable $Q^{(0)}$ and $R^{(0)}$ — even if these initial guesses are far from the truth, the resulting $W^{(0)}$ will produce a stable closed-loop matrix $\bar{F}$.

### 13.3 Pseudoinverse of $\Gamma$

When $\Gamma$ is tall and full column rank (i.e., $n_x > n_v$), the pseudoinverse is $\Gamma^{\dagger} = (\Gamma'\Gamma)^{-1}\Gamma'$. Use `numpy.linalg.pinv(Gamma)` with a sensible `rcond` parameter.

### 13.4 Minimal Polynomial Computation

Compute the eigenvalues of $\bar{F}$, then identify **distinct** eigenvalues (using a tolerance of $\sim 10^{-8}$ to merge near-duplicates). The minimal polynomial has degree $m$ equal to the number of distinct eigenvalues, and its coefficients are obtained from `numpy.poly(distinct_eigenvalues)`.

### 13.5 Baseline Comparisons Need Large $N$

If comparing against Mehra's or Belanger's methods, note that these older methods require $N \gtrsim 5000$ samples to produce stable gains. For smaller $N$, their $\bar{F}$ may be unstable, and those runs should be excluded from statistics.

---

## 14. Test Cases Summary

| Case | System | $n_x$ | $n_z$ | $n_v$ | $N$ | $M$ | MC Runs | $\lambda_Q$ | Key Feature |
|------|--------|--------|--------|--------|-----|-----|---------|-------------|-------------|
| 1 | Kinematic | 2 | 1 | 1 | 1000 | 10-100 | 100 | 0 | Vary $M$, study lag sensitivity |
| 2 | Neethling | 2 | 1 | 1 | 1000 | 100 | 100 | 0 | Easiest case, cond($\mathscr{I}$)=2.3 |
| 3 | Mehra 5-state | 5 | 2 | 3 | 10000 | 40 | 100 | 0 | Main benchmark, diagonal $Q$ and $R$ |
| 4 | Odelson 2-state | 2 | 1 | 1 | 1000 | 100 | 100 | 0.1 | Not fully observable, only detectable |
| 5 | Odelson 3-state | 3 | 1 | 1 | 1000 | 15 | 200 | 0.3 | Ill-conditioned, needs regularization |

**Recommended implementation order:** Start with Case 2 (easiest), then Case 1 (study $M$ sensitivity), then Case 3 (main benchmark), then Cases 4-5 (stress tests).
