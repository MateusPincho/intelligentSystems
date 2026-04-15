# Comprehensive Report: On the Identification of Noise Covariances and Adaptive Kalman Filtering

**Reference:** Zhang, L., Sidoti, D., Bienkowski, A., Pattipati, K. R., Bar-Shalom, Y., & Kleinman, D. L. (2020). *On the Identification of Noise Covariances and Adaptive Kalman Filtering: A New Look at a 50 Year-Old Problem*. IEEE Access, Vol. 8, pp. 59362-59388. DOI: 10.1109/ACCESS.2020.2982407

---

## 1. Introduction and Problem Statement

The Kalman filter (KF) is the optimal state estimator for linear dynamic systems driven by Gaussian white noise. However, its optimality requires **exact knowledge** of the process noise covariance $Q$ and measurement noise covariance $R$. In practice, these are almost never known exactly. Poorly chosen $Q$ and $R$ lead to suboptimal state estimates, biased innovations, and inconsistent filters.

This paper provides:
1. A **necessary and sufficient identifiability condition** for $Q$ and $R$ via a rank condition on a noise covariance identifiability matrix $\mathscr{I}$.
2. A **novel six-step iterative algorithm** that estimates $W$, $S$, $R$, $Q$, $\bar{P}$, and $P$ using a successive approximation approach with adaptive gradient descent.
3. **Five equivalent formulas (R1-R5)** for estimating $R$ using post-fit residuals.
4. **Iterative coupled estimation of $Q$ and $P$** with structural constraints enforcement.

---

## 2. System Model

### 2.1 Plant and Measurement Model

Consider the discrete-time linear dynamic system (Gauss-Markov):

$$x(k+1) = F x(k) + \Gamma v(k) \tag{1}$$

$$z(k) = H x(k) + w(k) \tag{2}$$

where:
- $x(k) \in \mathbb{R}^{n_x}$: state vector
- $z(k) \in \mathbb{R}^{n_z}$: measurement vector
- $v(k) \in \mathbb{R}^{n_v}$: process noise, $v(k) \sim \mathcal{N}(0, Q)$
- $w(k) \in \mathbb{R}^{n_z}$: measurement noise, $w(k) \sim \mathcal{N}(0, R)$
- $F \in \mathbb{R}^{n_x \times n_x}$: state transition matrix (known)
- $H \in \mathbb{R}^{n_z \times n_x}$: measurement matrix (known)
- $\Gamma \in \mathbb{R}^{n_x \times n_v}$: process noise gain matrix (known)
- $Q \in \mathbb{R}^{n_v \times n_v}$: process noise covariance (**unknown**, positive semi-definite)
- $R \in \mathbb{R}^{n_z \times n_z}$: measurement noise covariance (**unknown**, positive definite)

The noise sequences $v(k)$ and $w(k)$ are zero-mean, white, mutually independent, and independent of the initial state. The system is assumed to be observable and $(F, \Gamma Q^{1/2})$ controllable.

### 2.2 Standard Kalman Filter Equations

**Prediction (time update):**

$$\hat{x}(k+1|k) = F \hat{x}(k|k) \tag{3}$$

$$P(k+1|k) = F P(k|k) F' + \Gamma Q \Gamma' \tag{7}$$

**Innovation:**

$$\nu(k+1) = z(k+1) - H \hat{x}(k+1|k) \tag{4}$$

**Innovation covariance:**

$$S(k+1) = H P(k+1|k) H' + R \tag{7}$$

**Kalman gain:**

$$W(k+1) = P(k+1|k) H' S(k+1)^{-1} \tag{8}$$

**Update (measurement update):**

$$\hat{x}(k+1|k+1) = \hat{x}(k+1|k) + W(k+1) \nu(k+1) \tag{5}$$

**Updated state covariance:**

$$P(k+1|k+1) = P(k+1|k) - W(k+1) S(k+1) W(k+1)' \tag{9}$$

Or equivalently in **Joseph form** (more numerically stable):

$$P(k+1|k+1) = (I_{n_x} - W(k+1)H) P(k+1|k) (I_{n_x} - W(k+1)H)' + W(k+1) R\, W(k+1)' \tag{13}$$

---

## 3. Steady-State Kalman Filter

The six-step approach in this paper is designed for a **steady-state** Kalman filter, where the gain $W$ is constant over time.

### 3.1 Steady-State Prediction Covariance (DARE)

The steady-state prediction covariance $\bar{P}$ satisfies the **Discrete Algebraic Riccati Equation** (DARE):

$$\bar{P} = F[\bar{P} - \bar{P}H'(H\bar{P}H' + R)^{-1}H\bar{P}]F' + \Gamma Q \Gamma' \tag{10}$$

### 3.2 Steady-State Updated Covariance

The steady-state updated covariance $P$ satisfies (see Appendix A):

$$P = \bar{P} - W S W' \tag{12}$$

Or equivalently:

$$P = (I_{n_x} - WH)\bar{P}(I_{n_x} - WH)' + WRW' \tag{13}$$

It can also be computed as:

$$P = \left(\bar{P}^{-1} + H'R^{-1}H\right)^{-1} \tag{Appendix A: 220}$$

### 3.3 Steady-State Gain and Innovation Covariance

$$W = \bar{P}H'S^{-1} = PH'R^{-1} \tag{14}$$

$$S = E[\nu(k)\nu(k)'] = H\bar{P}H' + R \tag{15}$$

### 3.4 Closed-Loop Filter Matrix

$$\bar{F} = F(I_{n_x} - WH) \tag{16}$$

Note that $(I_{n_x} - WH)$ is invertible, but $\bar{F}$ need not be stable (eigenvalues inside the unit circle). However, for a stable filter, eigenvalues of $\bar{F}$ must be inside the unit circle.

### 3.5 Relationships Between $\bar{P}$, $P$, $Q$, and $S$

Three equivalent ways to compute $\Gamma Q \Gamma'$:

$$\textbf{Q1:} \quad \Gamma Q \Gamma' = F^{-1}\bar{P}(F^{-1})' + WSW' - \bar{P} \tag{119}$$

$$\textbf{Q2:} \quad \Gamma Q \Gamma' = P + WSW' - FPF' \tag{120}$$

$$\textbf{Q3:} \quad \Gamma Q \Gamma' = \bar{P} - F\bar{P}F' + FWRW'F' \tag{121}$$

And the key relationship:

$$\bar{P} = FPF' + \Gamma Q \Gamma' \tag{111}$$

$$P = \left((FPF' + \Gamma Q \Gamma')^{-1} + H'R^{-1}H\right)^{-1} \tag{116}$$

---

## 4. Identifiability of $Q$ and $R$

### 4.1 Minimal Polynomial of $\bar{F}$

Define the $m$-th order **minimal polynomial** of $\bar{F}$ as:

$$\sum_{i=0}^{m} a_i \bar{F}^{m-i} = 0, \quad a_0 = 1 \tag{21}$$

The minimal polynomial is the lowest-degree monic polynomial $p(\lambda)$ such that $p(\bar{F}) = 0$. It can be computed from the eigenvalues of $\bar{F}$: find the distinct eigenvalues and form $p(\lambda) = \prod_{j} (\lambda - \lambda_j)$.

### 4.2 Innovation Transformation

Using the minimal polynomial, the innovation $\nu(k)$ can be written as:

$$\nu(k) = H\bar{F}^m \tilde{x}(k-m|k-m-1) + \left\{ H \sum_{j=0}^{m-1} \bar{F}^{m-1-j} \left[\Gamma v(k-m+j) - FWw(k-m+j)\right] \right\} + w(k) \tag{22}$$

Define the **whiteness test statistic** $\xi(k)$ as:

$$\xi(k) = \sum_{i=0}^{m} a_i \nu(k-i) \tag{23}$$

When the minimal polynomial annihilates $\bar{F}$ (i.e., $\bar{F}$ is the optimal closed-loop matrix), $\xi(k)$ becomes a sum of two **moving average processes** driven by $v(k)$ and $w(k)$:

$$\xi(k) = \sum_{l=1}^{m} \mathscr{B}_l v(k-l) + \sum_{l=0}^{m} \mathscr{G}_l w(k-l) \tag{27}$$

where:

$$\mathscr{B}_l = H\left(\sum_{i=0}^{l-1} a_i \bar{F}^{l-i-1}\right) \Gamma \tag{28}$$

$$\mathscr{G}_l = a_l I_{n_z} - H\left(\sum_{i=0}^{l-1} a_i \bar{F}^{l-i-1}\right) FW \tag{29}$$

$$\mathscr{G}_0 = I_{n_z} \tag{30}$$

### 4.3 Lag Covariances $L_j$

Denoting $L_j = E[\xi(k)\xi(k-j)']$ for $j = 0, 1, 2, \ldots, m$:

$$L_j = \sum_{l=j+1}^{m} \mathscr{B}_l Q \mathscr{B}_{l-j}' + \sum_{l=j}^{m} \mathscr{G}_l R \mathscr{G}_{l-j}' \tag{31}$$

### 4.4 Identifiability Matrix $\mathscr{I}$

Exploiting the symmetry of $Q = [q_{ij}]$ ($n_v \times n_v$ positive semi-definite symmetric) and $R = [r_{ij}]$ ($n_z \times n_z$ positive definite symmetric), equation (31) can be rewritten as a linear system.

Using the vectorization convention:

$$\text{vec}(A) \triangleq [a_{11}, \ldots, a_{p1}, a_{12}, \ldots, a_{p2}, \ldots, a_{1n}, \ldots, a_{pn}]' \tag{35}$$

The **noise covariance identifiability matrix** $\mathscr{I}$ is of dimension $(m+1)n_z^2 \times \frac{1}{2}[n_v(n_v+1) + n_z(n_z+1)]$ and satisfies:

$$\mathscr{I} \begin{bmatrix} \text{vec}(Q) \\ \text{vec}(R) \end{bmatrix} = \begin{bmatrix} L_0 \\ L_1 \\ \vdots \\ L_m \end{bmatrix} \tag{36}$$

### 4.5 Algorithm 1: Construction of $\mathscr{I}$

The matrix $\mathscr{I}$ is constructed column by column from the $\mathscr{B}$ and $\mathscr{G}$ matrices. The algorithm iterates over:
- Lag index $j = 0, \ldots, m$
- For each row block corresponding to lag $j$:
  - Columns corresponding to $Q$: built from products $b_{i,l} b'_{i-j,l}$ (columns of $\mathscr{B}$)
  - Columns corresponding to $R$: built from products $g_{i,l} g'_{i-j,l}$ (columns of $\mathscr{G}$)

The symmetry of $Q$ and $R$ is exploited so that only the unique elements appear as unknowns.

**Detailed pseudocode:**

```
for j = 0 to m:
    r = j * n_z^2
    k = 0
    for l = 1 to n_v:                          # Q columns
        k += 1
        b = sum_{i=j+1}^{m} [b_{i,l} * b'_{i-j,l}]
        I(r+1 : r+n_z^2, k) = vec(b)
    for p = l+1 to n_v:                        # Q off-diagonal
        k += 1
        c_{j,l,l}(p) = [b_{i,l}*b'_{i-j,p} + b_{i,p}*b'_{i-j,l}]
        d = sum_{i=j+1}^{m} c_{j,l,l}(p)
        I(r+1 : r+n_z^2, k) = vec(d)
    for l = 1 to n_z:                          # R columns
        k += 1
        g = sum_{i=j}^{m} [g_{i,l} * g'_{i-j,l}]
        I(r+1 : r+n_z^2, k) = vec(g)
    for p = l+1 to n_z:                        # R off-diagonal
        k += 1
        h_{j,l,p}(i) = [g_{i,l}*g'_{i-j,p} + g_{i,p}*g'_{i-j,l}]
        f = sum_{i=j}^{m} h_{j,l,p}(i)
        I(r+1 : r+n_z^2, k) = vec(f)
```

### 4.6 Identifiability Condition

$$\text{rank}(\mathscr{I}) - n_R \geq n_Q \tag{37}$$

where $n_R$ is the number of unknowns in $R$ and $n_Q$ is the number of unknowns in $Q$.

Since $R$ is always estimable (because $\mathscr{G}_m$ is always invertible since $a_m \neq 0$ and $\bar{F}$ is invertible), the constraint is on whether there are enough independent measurements to also estimate $Q$.

**Key point:** The rank of $\mathscr{I}$ is independent of $W$, so for convenience one can examine the rank for $W = 0$.

---

## 5. Estimation of $W$ (Kalman Filter Gain)

### 5.1 Sample Autocovariances

Given an innovation sequence $\{\nu(k)\}_{k=1}^{N}$, compute $M$ sample autocovariance matrices:

$$\hat{C}(i) = \frac{1}{N-M} \sum_{j=1}^{N-M} \nu(j)\nu(j+i)' \quad \text{for } i = 0, 1, 2, \ldots, M-1 \tag{50}$$

### 5.2 Theoretical Autocovariance Structure

The theoretical autocovariance of the innovation sequence at lag $i \geq 1$ is:

$$C(i) = E[\nu(k)\nu(k-i)'] = H\bar{F}^{i-1}F[PH' - WC(0)] \tag{51}$$

At the **optimal** gain $W^*$, the innovations are white, meaning $C(i) = 0$ for $i \geq 1$.

### 5.3 Objective Function $J$

The objective function to minimize is the **normalized sum of off-zero-lag autocorrelations**:

$$J = \frac{1}{2} \text{tr}\left\{ \sum_{i=1}^{M-1} \left[\text{diag}(\hat{C}(0))\right]^{-1/2} \hat{C}(i)' \left[\text{diag}(\hat{C}(0))\right]^{-1} \hat{C}(i) \left[\text{diag}(\hat{C}(0))\right]^{-1/2} \right\} \tag{52}$$

where $\text{diag}(C)$ is the Hadamard product of $C$ with an identity matrix of the same dimension (i.e., keeping only the diagonal):

$$\text{diag}(C) = I \odot C \tag{53}$$

Using the notation:

$$\mathscr{E} = \left[\text{diag}(C(0))\right]^{-1/2} \tag{59}$$

$$\Psi = X - WC(0) \tag{57}$$

where $X = PH'$ (from eq. 58), the objective can be rewritten as:

$$J = \frac{1}{2} \text{tr}\left\{ \sum_{i=1}^{M-1} \Theta(i) X \mathscr{E}^2 X' \right\} \tag{54}$$

with:

$$\Theta(i) = \Phi(i)' \mathscr{E}^2 \Phi(i) \tag{55}$$

$$\Phi(i) = H\bar{F}^{i-1}F \tag{56}$$

### 5.4 Correlation Residual Matrix $X$

$X$ is obtained from the least-squares solution:

$$\begin{bmatrix} HF \\ H\bar{F}F \\ \vdots \\ H\bar{F}^{M-2}F \end{bmatrix} X = \begin{bmatrix} \hat{C}(1) \\ \hat{C}(2) \\ \vdots \\ \hat{C}(M-1) \end{bmatrix} \tag{62}$$

Using the pseudoinverse:

$$X = \begin{bmatrix} HF \\ H\bar{F}F \\ \vdots \\ H\bar{F}^{M-2}F \end{bmatrix}^{\dagger} \begin{bmatrix} \hat{C}(1) \\ \hat{C}(2) \\ \vdots \\ \hat{C}(M-1) \end{bmatrix} \tag{63}$$

where $A^{\dagger} = (A'A)^{-1}A'$ is the pseudoinverse.

### 5.5 Gradient of $J$ with Respect to $W$

The gradient is (derived in Appendix E):

$$\nabla_W J = -\sum_{i=1}^{M-1} \Theta(i)[\Psi - WC(0)] \mathscr{E}^2 C(0) - F' Z F X \tag{60}$$

$$- \sum_{\ell=0}^{i-2} \left[C(\ell+1) \mathscr{E}^2 C(i) \mathscr{E}^2 H \bar{F}^{i-\ell-2}\right]$$

More precisely, from Appendix E (eq. 265):

$$\nabla_W J = -\left[\sum_{i=1}^{M-1} \Theta(i) X \mathscr{E}^2 C(0) + F' Z F X \right] \tag{60, simplified}$$

where $Z$ is the solution to the **discrete Lyapunov equation**:

$$Z = \bar{F}' Z \bar{F} + \frac{1}{2} \sum_{i=1}^{M-1} \left[\Theta(i)(\Psi - WC(0)) \mathscr{E}^2 H + H' \mathscr{E}^2 (\Psi' - C(0)W') \Theta(i)\right] (\bar{F}')^{i-1} \tag{61 (implicit)}$$

This can be solved using the standard discrete Lyapunov solver since $\bar{F}$ is stable. The full expansion is:

$$Z = \sum_{b=0}^{\infty} (\bar{F}')^b \left[\frac{1}{2} \sum_{i=1}^{M-1} \left(\Theta(i)(\Psi - WC(0))\mathscr{E}^2 H + H'\mathscr{E}^2(\Psi' - C(0)W')\Theta(i)\right)\right] \bar{F}^b \tag{264}$$

For **practical implementation**, the gradient simplifies to (from eq. 265):

$$\nabla_W J = -\left[\sum_{i=1}^{M-1} \Theta(i) X \mathscr{E}^2 \hat{C}(0) + F' Z F X\right]$$

where:
- Replace $C(0)$ with $\hat{C}(0)$ (the sample estimate)
- $X$ is from eq. (63)
- $\Psi = X - W\hat{C}(0)$
- $Z$ solves the Lyapunov equation (61) with sample quantities

### 5.6 For ill-conditioned systems, regularization

For ill-conditioned systems, a regularization term $\lambda_W \text{tr}(WW')$ can be added to convexify the objective. The gradient becomes:

$$\nabla_W J = -\sum_{i=1}^{M-1} \Theta(i)[\Psi - WC(0)] \mathscr{E}^2 C(0) - F' Z F X - \sum_{\ell=0}^{i-2} [C(\ell+1) \mathscr{E}^2 C(i) \mathscr{E}^2 H \bar{F}^{i-\ell-2}] \tag{60}$$

(Note: the regularization is mentioned on p. 59369 but its detailed gradient form is implied.)

---

## 6. Estimation of $R$

### 6.1 Post-Fit Residuals

Given the optimal steady-state gain $W$ and innovation sequence $\nu(k)$, the **post-fit residual** is:

$$\mu(k) = z(k) - H\hat{x}(k|k) \tag{65}$$

$$= (I_{n_z} - HW)\nu(k) \tag{66}$$

### 6.2 Joint Covariance Structure

The joint covariance of innovations and post-fit residuals is:

$$\text{Cov}\begin{pmatrix} \nu(k) \\ \mu(k) \end{pmatrix} = \begin{bmatrix} S & R \\ R & R - HPH' \end{bmatrix} \tag{67}$$

The post-fit residual covariance is:

$$G = E[\mu(k)\mu(k)'] = R - HPH' \tag{74}$$

Using (14), $G = R - HPH'$ and $S - R = H\bar{P}H'$, we get:

$$G = R S^{-1} R \tag{85}$$

This is the key relationship used in **R3**.

### 6.3 Five Formulas for Estimating $R$

All five are **theoretically equivalent** but differ numerically:

$$\textbf{R1:} \quad R = (I_{n_z} - HW)S \tag{76}$$

$$\textbf{R2:} \quad R = \frac{1}{2}\left[E[\mu(k)\nu(k)'] + E[\nu(k)\mu(k)']\right] \tag{77}$$

$$\textbf{R3:} \quad G = R S^{-1} R \tag{78}$$

$$\textbf{R4:} \quad R = \frac{1}{2}\left[G + S - HWSW'H'\right] \tag{79}$$

$$\textbf{R5:} \quad R = \frac{1}{2}\left[G(I_{n_z} - W'H')^{-1} + (I_{n_z} - HW)^{-1}G\right] \tag{80}$$

**In practice**, with sample estimates:
- $S \approx \hat{C}(0)$ (sample innovation covariance)
- $G \approx \hat{G} = \frac{1}{N}\sum_{k=1}^{N} \mu(k)\mu(k)'$ (sample post-fit residual covariance)

### 6.4 Why R3 is Recommended

**R3** is the recommended method because:
1. It guarantees **positive definiteness** of the estimated $R$ (since it is equivalent to a continuous-time algebraic Riccati equation)
2. The equation $G = RS^{-1}R$ with $S = \hat{C}(0)$ and $G = \hat{G}$ always has a positive definite solution

### 6.5 Solving R3 via Cholesky Decomposition and Eigendecomposition (Appendix F)

**Step 1:** Cholesky decomposition of $S^{-1}$:

$$S^{-1} = \mathscr{L}\mathscr{L}' \tag{266}$$

**Step 2:** Form the product $\mathscr{L}'RS^{-1}R\mathscr{L}$ and note that:

$$\mathscr{L}' R S^{-1} R \mathscr{L} = (\mathscr{L}' R \mathscr{L})^2 = \mathscr{L}' G \mathscr{L} \tag{267}$$

**Step 3:** Eigendecompose $\mathscr{L}' G \mathscr{L}$:

$$\mathscr{L}' G \mathscr{L} = U \Lambda U' \tag{268}$$

where $\Lambda$ is diagonal with non-negative eigenvalues.

**Step 4:** Take the matrix square root:

$$\mathscr{L}' R \mathscr{L} = U \Lambda^{1/2} U' \tag{269}$$

**Step 5:** Recover $R$:

$$R = (\mathscr{L}')^{-1} U \Lambda^{1/2} U' \mathscr{L}^{-1} \tag{270}$$

### 6.6 Solving R3 via Simultaneous Diagonalization (Appendix G)

**Step 1:** Eigendecompose $S^{-1}$:

$$S^{-1} = U_1 \Lambda_1 U_1' = (U_1 \Lambda_1^{1/2} U_1')^2 \tag{271-272}$$

**Step 2:** Form and eigendecompose:

$$S^{-1/2} G S^{-1/2} = (S^{-1/2} R S^{-1/2})^2 \tag{273}$$

$$U_1 \Lambda_1^{1/2} U_1' G U_1 \Lambda_1^{1/2} U_1' = (U_2 \Lambda_2^{1/2} U_2')^2 \tag{274}$$

which equals $(U_1 \Lambda_1^{1/2} U_1' R U_1 \Lambda_1^{1/2} U_1')^2$.

**Step 3:** Recover $R$:

$$R = U_1 \Lambda_1^{-1/2} U_1' U_2 \Lambda_2^{1/2} U_2' U_1 \Lambda_1^{-1/2} U_1' \tag{276}$$

### 6.7 Diagonal $R$

When $R$ is known to be diagonal, solve the **least squares problem**:

$$\min_{R \geq 0} \|X - R\|_F^2 \tag{95}$$

where $X$ is the full $R$ estimate. The solution is simply to keep only the diagonal elements:

$$\hat{R} = \text{diag}(\hat{R}_{\text{full}})$$

This can also be interpreted as applying a **mask matrix** to enforce structural constraints.

---

## 7. Estimation of $Q$ and $P$

### 7.1 Overview

Unlike $R$, the estimation of $Q$ and $P$ (or $\bar{P}$) requires an **iterative procedure** because they are coupled through the Riccati equation. Let $t = 0, 1, \ldots$ denote the iteration index for $Q$ and $\ell = 0, 1, \ldots$ denote the inner iteration index for $P$.

### 7.2 Initialization

Initialize using the **Wiener process solution** (eq. 165):

$$\Gamma Q^{(0)} \Gamma' = W S W' \tag{165}$$

This gives:

$$Q^{(0)} = \Gamma^{\dagger} (W S W') (\Gamma')^{\dagger}$$

Initialize $P^{(0)}$ by solving the **Lyapunov equation** (eq. 122):

$$P^{(0)} = \bar{F} P^{(0)} \bar{F}' + WRW' + (I_{n_x} - WH) \Gamma Q^{(0)} \Gamma' (I_{n_x} - WH)' \tag{122}$$

This is a discrete Lyapunov equation in $P^{(0)}$ which can be solved using `scipy.linalg.solve_discrete_lyapunov`.

### 7.3 Inner Loop: Iterating $P$ (index $\ell$)

For a given $Q^{(t)}$, iterate to convergence:

$$P^{(\ell+1)} = \left[(FP^{(\ell)}F' + \Gamma Q^{(t)} \Gamma')^{-1} + H'R^{-1}H\right]^{-1} \tag{123}$$

This is derived from eq. (116). Iterate until $\|P^{(\ell+1)} - P^{(\ell)}\| < \text{tol}$.

### 7.4 Updating $Q$ (index $t$)

Given the converged $P$, define:

$$D^{(t+1)} = P + WSW' - FPF' \tag{124}$$

Then update $Q$:

$$Q^{(t+1)} = \Gamma^{\dagger} D^{(t+1)} (\Gamma')^{\dagger} \tag{125}$$

### 7.5 Structural Constraints via Mask Matrix

A **mask matrix** $A$ enforces structural constraints (e.g., diagonality of $Q$):

$$Q^{(t+1)} = A \odot Q^{(t+1)} \tag{126}$$

where $\odot$ is the Hadamard (element-wise) product and $A$ is a binary matrix with 1s at positions corresponding to the known structure (e.g., identity matrix for diagonal $Q$).

### 7.6 Regularization for Ill-Conditioned Systems

For ill-conditioned systems, a regularization parameter $\lambda_Q$ can be used:

$$Q^{(t+1)} = \Gamma^{\dagger} \left[D^{(t+1)} + \lambda_Q I_{n_x}\right] (\Gamma')^{\dagger} \tag{127}$$

After the mask is applied:

$$Q^{(t+1)} = A \odot Q^{(t+1)}$$

Then set $\ell = 0$ and recompute $P$ using (123) with the new $Q^{(t+1)}$. Repeat until $\|Q^{(t+1)} - Q^{(t)}\| < \text{tol}$.

### 7.7 Computing $\bar{P}$

After $Q$ and $P$ converge, compute the prediction covariance:

$$\bar{P} = FPF' + \Gamma Q \Gamma' \tag{111}$$

or equivalently from (112) or (113).

---

## 8. The Complete Six-Step Algorithm

### Step 1: Run the Kalman Filter

Start with iteration $r = 0$ and initialize with a $W^{(0)}$ obtained from DARE to stabilize the system. Execute the **steady-state Kalman filter** with constant gain $W^{(r)}$ for $k = 1, 2, \ldots, N$:

$$\hat{x}^{(r)}(k+1|k) = F\hat{x}^{(r)}(k|k) \tag{128}$$

$$\nu^{(r)}(k+1) = z(k+1) - H\hat{x}^{(r)}(k+1|k) \tag{129}$$

$$\hat{x}^{(r)}(k+1|k+1) = \hat{x}^{(r)}(k+1|k) + W^{(r)}\nu^{(r)}(k+1) \tag{130}$$

$$\mu^{(r)}(k+1) = z(k+1) - H\hat{x}^{(r)}(k+1|k+1) \tag{131}$$

**Implementation notes:**
- Initialize $\hat{x}^{(r)}(0|0) = 0$ (or draw from $\mathcal{N}(0, P)$ for simulation)
- The gain $W^{(r)}$ is **constant** throughout the entire pass
- Store both $\nu^{(r)}(k)$ and $\mu^{(r)}(k)$ for all $k$

### Step 2: Compute Sample Autocovariances

Compute $M$ sample autocovariance matrices from the innovation sequence:

$$\hat{C}(i) = \frac{1}{N-M} \sum_{j=1}^{N-M} \nu(j)\nu(j+i)' \quad \text{for } i = 0, 1, \ldots, M-1 \tag{50}$$

Choose $M \geq n_x$. Typical values: $M = 40$-$100$.

### Step 3: Update $W$ via Adaptive Gradient Descent

#### 3.1 Termination Conditions

Check five conditions. If **any** is met, terminate the gradient descent:

**Condition 1:** The change in gain is small:

$$\delta_W = \|\Delta W ./ (W^{(r)} + \epsilon_W)\| < \zeta_W \tag{132-133}$$

where $\Delta W = W^{(r+1)} - W^{(r)}$, $./$  is element-wise division, $\epsilon_W$ is a small constant to protect against zeros, and $\|\cdot\|$ is the Euclidean (Frobenius) norm.

**Condition 2:** The gradient norm is small:

$$\|\nabla_W J\|_2 < \zeta_\Delta \tag{134}$$

**Condition 3:** The objective value is small:

$$J < \zeta_J$$

**Condition 4:** The objective has stopped improving for a specified "patience" number of epochs.

**Condition 5:** Maximum number of iterations $n_L$ reached.

#### 3.2 Gain Update

If none of the termination conditions are met, update:

$$W^{(r+1)} = W^{(r)} - \alpha^{(r)} \nabla_W J \tag{135}$$

#### 3.3 Step Size Initialization (Bold Driver)

The initial step size is:

$$\alpha^{(0)} = \min\left(c \left(\frac{N}{N_s}\right)^\beta, \bar{c}\right) \tag{136}$$

where:
- $c = 0.01$ is a base constant
- $N_s = N$ is a hyperparameter (number of observations)
- $\beta = 2$ is a positive constant
- $\bar{c} = c_{\max} = 0.2$ is the maximum step size

With $N_s = N$ and $\beta = 2$, the initial step size simplifies to $\alpha^{(0)} = \min(c, \bar{c}) = c = 0.01$.

#### 3.4 Step Size Adaptation (Bold Driver Method)

At each iteration $r$, compare $J^{(r)}$ to $J^{(r-1)}$:

$$\alpha^{(r)} = \begin{cases} 0.5\,\alpha^{(r-1)} & \text{if } J^{(r)} > J^{(r-1)} \\ \max(1.1\,\alpha^{(r-1)},\ \bar{c}) & \text{otherwise} \end{cases} \tag{137}$$

where $\bar{c}$ is the maximum step size:

$$\bar{c} = \min\left(\left(\frac{N}{N_s}\right)^\beta,\ c_{\max}\right) \tag{138}$$

and $c_{\max}$ is a positive constant between 0 and 1.

#### 3.5 Best Gain Selection

At each iteration, save $W^{(r)}$ and $J^{(r)}$ if $J^{(r)} \leq J^{(r-1)}$. Upon termination, select the gain that achieved the minimum $J$:

$$W = \arg\min_r J^{(r)} \tag{139}$$

### Step 4: Estimate $R$

Using the converged $W$ from Step 3 and $S = \hat{C}(0)$:
1. Compute the sample post-fit residual covariance: $\hat{G} = \frac{1}{N}\sum_{k=1}^{N} \mu(k)\mu(k)'$
2. Estimate $R$ using **R3**: solve $\hat{G} = R S^{-1} R$ via Cholesky + eigendecomposition (Appendix F) or simultaneous diagonalization (Appendix G)
3. If $R$ is known to be diagonal, keep only $\text{diag}(\hat{R})$

### Step 5: Estimate $Q$ and $P$

Using the estimated $R$ from Step 4:
1. Initialize $\Gamma Q^{(0)} \Gamma' = WSW'$ (Wiener process solution, eq. 165)
2. Solve the Lyapunov equation (122) for $P^{(0)}$
3. **Inner iteration** (index $\ell$): compute $P^{(\ell+1)}$ from (123) until convergence
4. **Update** $Q^{(t+1)}$ from (124)-(125)
5. Apply mask: $Q^{(t+1)} = A \odot Q^{(t+1)}$ (eq. 126)
6. For ill-conditioned systems, add regularization $\lambda_Q$ (eq. 127)
7. Repeat steps 3-6 until $\|Q^{(t+1)} - Q^{(t)}\| < \text{tol}$
8. Compute $\bar{P} = FPF' + \Gamma Q \Gamma'$ (eq. 111)

### Step 6: Outer Successive Approximation

1. Reinitialize: solve DARE with the newly estimated $(Q, R)$ to get a new $W^{(0)}$
2. Repeat Steps 1-5
3. Track the best $J^{(r)}$ across outer loops
4. Terminate when the improvement in $J$ between outer loops is below $\zeta_J$, or after 20 outer iterations
5. Select the $W$ corresponding to the minimum $J$ across all outer loops

---

## 9. Special Cases

### 9.1 Case A: Wiener Process ($F = I_{n_x}$, $H = I_{n_z}$, $n_x = n_z$)

When both $F$ and $H$ are identity matrices:

**Optimal gain from sample data:**

Define:

$$\xi(k) = z(k) - z(k-1) = \hat{x}(k|k-1) + v(k) - \hat{x}(k-1|k-2) + w(k) - w(k-1) \tag{143}$$

$$= v(k) + (W - I_{n_z})v(k-1) \tag{145}$$

Compute sample lag covariances:

$$L_0 = E[\xi(k)\xi(k)'] = S + (W - I_{n_z})S(W - I_{n_z})' \tag{146}$$

$$L_1 = E[\xi(k)\xi(k-1)'] = (W - I_{n_z})S \tag{147}$$

The optimal gain is:

$$W = I_{n_z} + L_1 S^{-1} \tag{148}$$

And $S$ is found by solving the **positive definite Riccati equation**:

$$L_0 = S + L_1 S^{-1} L_1' \tag{150}$$

**Closed-form estimates:**

$$\bar{P} = WS \tag{164}$$

$$Q = WSW' \tag{165}$$

$$P = \bar{P} - WSW' \tag{167 implies: } P = WS - WSW'$$

### 9.2 Case B: $H = I_{n_x}$ (General $F$)

When only $H$ is the identity:

Define $\xi(k) = z(k) - Fz(k-1) = \nu(k) - \bar{F}\nu(k-1)$ (eq. 175-177)

$$L_0 = \Gamma Q \Gamma' + R + FRF' \tag{185}$$

$$L_1 = -\bar{F}S \tag{180}$$

From $L_1$, solve for $S$:

$$S + L_1 S^{-1} L_1' = L_0 \tag{181}$$

Then:

$$W = I_{n_z} + F^{-1}L_1 S^{-1} \tag{182}$$

And $R$ is estimated via R3 ($G = RS^{-1}R$) and:

$$\Gamma Q \Gamma' = S + FGF' - (R + FRF') \tag{186-187}$$

---

## 10. Gradient Computation Details (Appendix E)

The full derivation of $\nabla_W J$ proceeds as follows:

### 10.1 Differential of $J$

$$\delta J = \frac{1}{2} \text{tr}\left\{\sum_{i=1}^{M-1} [\delta\Theta(i)\Omega + \Theta(i)\delta\Omega]\right\} \tag{251}$$

where:

$$\Omega = [\Psi - WC(0)] \mathscr{E}^2 [\Psi' - C(0)W'] \tag{252}$$

### 10.2 Structure of the Sum

After extensive manipulation (eqs. 253-265):

$$\frac{1}{2}\text{tr}\left(\sum_{i=1}^{M-1}\Theta(i)\delta\Omega\right) = -\text{tr}\left\{-\delta W' \left[\sum_{i=1}^{M-1} \Theta(i) X \mathscr{E}^2 \hat{C}(0)\right]\right\}$$

$$-\text{tr}\left\{[F\delta W(\Psi' - C(0)W')F' + F(\Psi - WC(0))\delta W' F'] Z\right\} \tag{263}$$

where $Z$ satisfies the Lyapunov equation:

$$Z = \sum_{b=0}^{\infty} (\bar{F}')^b \left[\frac{1}{2}\sum_{i=1}^{M-1} \left(\Theta(i)(\Psi - WC(0))\mathscr{E}^2 H + H'\mathscr{E}^2(\Psi' - C(0)W')\Theta(i)\right)\right](\bar{F})^b \tag{264}$$

### 10.3 Final Gradient

Combining all terms:

$$\nabla_W J = -\left[\sum_{i=1}^{M-1}\Theta(i) X \mathscr{E}^2 \hat{C}(0) + F' Z F X\right] \tag{265 → 60}$$

---

## 11. Consistency Check: Normalized Innovation Squared (NIS)

The averaged NIS over all Monte Carlo runs is:

$$\bar{\varepsilon}(k) = \frac{1}{n_{MC}} \sum_{i=1}^{n_{MC}} \nu(k)' S^{-1} \nu(k) \tag{191}$$

For a consistent filter, $\bar{\varepsilon}(k)$ should have mean $\approx n_z$ (or $\approx 1$ if normalized by $n_z$). The paper plots $\bar{\varepsilon}(k) / n_z$ and checks whether it lies within the 95% confidence band.

---

## 12. Hyperparameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| $\zeta_J$ | $10^{-6}$ | Objective function convergence tolerance |
| $\zeta_W$ | $10^{-6}$ | Gain change convergence tolerance |
| $\zeta_\Delta$ | $10^{-6}$ | Gradient norm convergence tolerance |
| $c$ | $0.01$ | Initial step size base constant |
| $c_{\max}$ | $0.2$ | Maximum step size |
| $\beta$ | $2$ | Step-size / sample-size exponent |
| patience | $5$ | Epochs of no $J$ improvement before stopping |
| $n_L$ | $100$-$500$ | Maximum inner iterations (gradient descent on $W$) |
| Outer-loop limit | $20$ | Maximum outer successive approximation iterations |
| $\lambda_Q$ | $0.1$-$0.3$ | Regularization parameter for $Q$ (ill-conditioned cases) |
| $M$ | $\geq n_x$, typically $40$-$100$ | Number of lags in sample autocovariance |
| $N$ | $1000$-$10000$ | Number of observed samples |
| $N_s$ | $N$ | Hyperparameter for step size initialization |

---

## 13. Numerical Test Cases

### Case 1: Second-Order Kinematic System

$$F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \quad \Gamma = \begin{bmatrix} \frac{T^2}{2} \\ T \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 \end{bmatrix} \tag{192-193}$$

with $T = 0.1$ (sampling period).

**True values:**

$$E[v(k)v(j)'] = 0.0025\,\delta_{kj}, \quad E[w(k)w(j)'] = 0.01\,\delta_{kj} \tag{194-195}$$

So $Q = 0.0025$ (scalar), $R = 0.01$ (scalar).

**Simulation:** $N = 1000$, $n_L = 100$, $N_s = 1000$. Vary $M \in \{10, 20, 30, 40, 50, 100\}$.

**Initial gain:** Obtained from DARE with $Q^{(0)} = 0.1$, $R^{(0)} = 0.1$:

$$W^{(0)} = \begin{bmatrix} 0.1319 \\ 0.0932 \end{bmatrix} \tag{197}$$

### Case 2: Neethling System

$$F = \begin{bmatrix} 0.8 & 1 \\ -0.4 & 0 \end{bmatrix}, \quad \Gamma = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 \end{bmatrix} \tag{198-199}$$

**True values:**

$$E[v(k)v(j)'] = \delta_{kj}, \quad E[w(k)w(j)'] = \delta_{kj} \tag{200-201}$$

So $Q = 1$ (scalar), $R = 1$ (scalar).

**Simulation:** $N = 1000$, $M = 100$, $n_L = 100$, $N_s = 1000$.

**Initial gain:**

$$W^{(0)} = \begin{bmatrix} 0.9 \\ 0.5 \end{bmatrix} \tag{203}$$

**Identifiability:** $\text{rank}(\mathscr{I}) = 2$, condition number of $\mathscr{I}$ is 2.3. Two unknowns ($Q$ and $R$), identifiable.

### Case 3: Mehra/Belanger 5-State System

$$F = \begin{bmatrix} 0.75 & -1.74 & -0.3 & 0 & -0.15 \\ 0.09 & 0.91 & -0.0015 & 0 & -0.008 \\ 0 & 0 & 0.95 & 0 & 0 \\ 0 & 0 & 0 & 0.55 & 0 \\ 0 & 0 & 0 & 0 & 0.905 \end{bmatrix} \tag{204}$$

$$\Gamma = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 24.64 & 0 & 0 \\ 0 & 0.835 & 0 \\ 0 & 0 & 1.83 \end{bmatrix} \tag{205}$$

$$H = \begin{bmatrix} 1 & 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 & 0 \end{bmatrix} \tag{206}$$

**True values:**

$$Q = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad R = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \tag{207}$$

Both $Q$ and $R$ are diagonal.

**Simulation:** $N = 10000$, $n_L = 500$, $N_s = 10000$, $M = 40$.

**Initial gain:** From DARE with:

$$Q^{(0)} = \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.5 & 0 \\ 0 & 0 & 0.75 \end{bmatrix}, \quad R^{(0)} = \begin{bmatrix} 0.4 & 0 \\ 0 & 0.7 \end{bmatrix} \tag{208-209}$$

**Identifiability:** $\text{rank}(\mathscr{I}) = 5$, condition number 808. Five unknowns (3 diagonal $Q$ + 2 diagonal $R$).

### Case 4: Odelson's Detectable System

$$F = \begin{bmatrix} 0.1 & 0 \\ 0 & 0.2 \end{bmatrix}, \quad \Gamma = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 \end{bmatrix} \tag{210-211}$$

**True values:**

$$E[v(k)v(j)'] = 0.5\,\delta_{kj}, \quad E[w(k)w(j)'] = \delta_{kj} \tag{212-213}$$

$Q = 0.5$, $R = 1$. **Note:** The system is **not fully observable** (the second state is unobservable from $z$), but it is **detectable**.

**Simulation:** $N = 1000$, $n_L = 100$, $N_s = 1000$, $M = 100$, $\lambda_Q = 0.1$.

**Initial gain:** From DARE with $R^{(0)} = 0.2$, $Q^{(0)} = 0.4$.

**Identifiability:** $\text{rank}(\mathscr{I}) = 2$, condition number 23.4. Two unknowns ($Q$ and $R$).

### Case 5: Odelson's 3-State Ill-Conditioned System

$$F = \begin{bmatrix} 0.1 & 0 & 0.1 \\ 0 & 0.2 & 0 \\ 0 & 0 & 0.3 \end{bmatrix}, \quad \Gamma = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad H = \begin{bmatrix} 0.1 & 0.2 & 0 \end{bmatrix} \tag{215-216}$$

**True values:**

$$E[v(k)v(j)'] = 0.5\,\delta_{kj}, \quad E[w(k)w(j)'] = 0.1\,\delta_{kj} \tag{217-218}$$

$Q = 0.5$, $R = 0.1$.

**Simulation:** $N = 1000$, **200 Monte Carlo runs**, $n_L = 100$, $N_s = 1000$, $M = 15$, $\lambda_Q = 0.3$.

**Initial gain:** From DARE with $R^{(0)} = 0.1$, $Q^{(0)} = 0.5$.

**Identifiability:** $\text{rank}(\mathscr{I}) = 2$, condition number 36.4. Two unknowns ($Q$ and $R$).

---

## 14. Monte Carlo Simulation Procedure

For each test case:

1. **Generate data:** Simulate the true system (1)-(2) with the true $Q$ and $R$ for $N$ time steps.
2. **Run the six-step algorithm** to estimate $W$, $S$, $R$, $Q$, $\bar{P}$, $P$.
3. **Repeat** for $n_{MC}$ Monte Carlo runs (100 for Cases 1-4, 200 for Case 5).
4. **Report:**
   - Mean of each estimated parameter
   - 95% highest probability interval $[\underline{r}, \bar{r}]$
   - RMSE of each estimated parameter
   - Averaged NIS $\bar{\varepsilon}(k)$ over MC runs with 95% chi-squared confidence band

### 14.1 RMSE Computation

For a scalar parameter $\theta$ with true value $\theta^*$ and $n_{MC}$ estimates $\hat{\theta}_i$:

$$\text{RMSE} = \sqrt{\frac{1}{n_{MC}} \sum_{i=1}^{n_{MC}} (\hat{\theta}_i - \theta^*)^2}$$

### 14.2 NIS Computation

$$\bar{\varepsilon}(k) = \frac{1}{n_{MC}} \sum_{i=1}^{n_{MC}} \nu_i(k)' S_i^{-1} \nu_i(k) \tag{191}$$

For a consistent filter, $\bar{\varepsilon}(k) \approx n_z$.

---

## 15. Summary of Key Implementation Steps

1. **Build infrastructure:** `LinearSystem` dataclass, DARE solver, closed-loop $\bar{F}$ helper, KF runner.
2. **Implement identifiability check:** Compute minimal polynomial coefficients, build $\mathscr{B}$/$\mathscr{G}$ matrices, construct $\mathscr{I}$, verify rank condition.
3. **Implement Step 3 (W estimation):** Sample autocovariances, objective $J$, gradient $\nabla_W J$ (via $\Theta$, $\Phi$, $\mathscr{E}$, $\Psi$, $X$, $Z$), bold-driver step size.
4. **Implement Step 4 (R estimation):** Post-fit residual covariance $\hat{G}$, solve R3 via Cholesky + eigendecomposition.
5. **Implement Step 5 (Q/P estimation):** Wiener initialization, Lyapunov solve for $P^{(0)}$, inner $P$ iteration, $Q$ update with mask and $\lambda_Q$.
6. **Implement Step 6 (outer loop):** Successive approximation with DARE reinitialization.
7. **Implement metrics:** RMSE, 95% PI, NIS.
8. **Run test cases** and compare to Tables 3-10 in the paper.
