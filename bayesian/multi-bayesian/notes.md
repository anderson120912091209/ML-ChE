# Machine Learning Notes — Gaussian Processes & Bayesian Optimization

> These notes build from scratch. Read top to bottom.

---

## RBF Kernel (quick recap)

$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

- $x$ and $x'$ are two data points (each a vector of all features)
- Output is between 0 and 1 — measures similarity
- Close points → similarity near 1
- Far points → similarity near 0
- $\sigma$ controls how quickly similarity drops off with distance

This kernel is the backbone of Gaussian Processes. Keep it in mind.

---

---

# Part 1 — Gaussian Processes

---

## 1.1 What problem are we solving?

You're a chem eng running reactor experiments. Each experiment costs time and money. You want to find the conditions (temperature, pressure, concentration) that maximize yield — but you can't run thousands of experiments.

You have maybe 10-20 data points. You need to:
1. Predict yield at conditions you haven't tried
2. Know **how confident** you are in that prediction
3. Decide **where to experiment next** to learn the most

Linear regression gives you (1) but not (2) or (3). A Gaussian Process gives you all three.

---

## 1.2 The core idea — a distribution over functions

Normal regression fits **one function** to your data. You get one line, one curve, one answer.

A Gaussian Process doesn't commit to one function. Instead it maintains a **distribution over all possible functions** that are consistent with your data.

Think of it this way:

```
Before you have any data:
  - infinitely many functions are possible
  - GP assigns probability to each one
  - functions that are smooth and well-behaved get higher probability
    (this is the prior — your assumption before seeing data)

After you observe some data:
  - most functions get ruled out (they don't pass through your data points)
  - only functions consistent with the data remain
  - GP updates its probability distribution
    (this is the posterior — your belief after seeing data)
```

When you ask "what is the yield at temp=380?", the GP looks at all surviving functions and reads off their values at that point. If all surviving functions agree → low uncertainty. If they spread out → high uncertainty.

---

## 1.3 Why is it called "Gaussian"?

At any single input point $x$, the GP's prediction is a **Gaussian (normal) distribution** — a bell curve with a mean and variance.

$$f(x) \sim \mathcal{N}(\mu(x),\ \sigma^2(x))$$

- $\mu(x)$ = your best prediction (the mean of the bell curve)
- $\sigma^2(x)$ = your uncertainty (the width of the bell curve)

At a point with lots of nearby data → narrow bell → confident prediction.
At a point far from all data → wide bell → uncertain prediction.

This is extremely useful in engineering. You don't just get a number, you get a number with an honest uncertainty attached.

---

## 1.4 Where does the kernel come in?

The kernel defines the **structure of your prior** — before you see any data, what do you assume functions look like?

With the RBF kernel you're saying:

> "I believe the true function is smooth. If I know the output at temp=350, then temp=351 should give a similar output. temp=400 somewhat similar. temp=600 might be completely different."

The kernel literally encodes this assumption mathematically. Two points with high kernel similarity will have correlated predictions — if one goes up, the other tends to go up too.

Different kernels = different assumptions about the function:
- **RBF** → smooth, infinitely differentiable function
- **Matérn** → rougher, more realistic for physical systems
- **Periodic** → function repeats (useful for seasonal data)
- **Linear** → assumes linear relationships

Choosing the right kernel is like choosing what kind of function you think nature is using.

---

## 1.5 The covariance matrix — the heart of a GP

When you have $n$ training points, the GP builds an $n \times n$ matrix called the **covariance matrix** $K$:

$$K_{ij} = k(x_i, x_j)$$

Each entry is the kernel value between point $i$ and point $j$ — how similar are those two data points.

This matrix encodes the entire relationship structure of your data. It tells the GP:
- Which points are similar to each other
- How much information each point gives you about its neighbors
- How uncertainty should spread across input space

When you ask for a prediction at a new point $x^*$, the GP computes:
- $k(x^*, x_i)$ for all training points — how similar is the new point to each training point
- Uses this to weight the training outputs
- Points that are similar to $x^*$ get high weight
- Points that are far from $x^*$ get low weight

This is exactly the kernel similarity idea from before, now used for prediction.

---

## 1.6 The prediction equations (what's actually happening)

Given:
- Training inputs $X = \{x_1, \ldots, x_n\}$
- Training outputs $y = \{y_1, \ldots, y_n\}$
- New test point $x^*$

The GP prediction is:

$$\mu(x^*) = k(x^*, X) \cdot K^{-1} \cdot y$$

$$\sigma^2(x^*) = k(x^*, x^*) - k(x^*, X) \cdot K^{-1} \cdot k(X, x^*)$$

Don't panic at the equations. Here's what they mean in plain English:

**Mean prediction $\mu(x^*)$:**
- $k(x^*, X)$ = vector of similarities between $x^*$ and all training points
- $K^{-1} y$ = a weighted version of your training outputs
- Multiply them together → you get a weighted average of training outputs, where similar training points get more weight

**Uncertainty $\sigma^2(x^*)$:**
- Start with maximum possible uncertainty at $x^*$: $k(x^*, x^*)$
- Subtract how much certainty the training data gives you
- Far from all data → subtract little → high uncertainty remains
- Close to training data → subtract a lot → low uncertainty remains

---

## 1.7 A concrete chem eng example

You run 5 reactor experiments:

```
temp:  300   350   380   400   420
yield: 0.62  0.78  0.85  0.81  0.72
```

You want to predict yield at temp=365 (you haven't tried this).

The GP:
1. Computes similarity between temp=365 and each training point
   - 365 vs 350: very similar (close together)
   - 365 vs 380: very similar (close together)
   - 365 vs 300: less similar (far away)
   - 365 vs 420: less similar (far away)

2. Weighted average: mostly influenced by the 350 and 380 points
   - yield at 350 = 0.78, yield at 380 = 0.85
   - prediction ≈ somewhere between these → maybe 0.82

3. Uncertainty: small, because two nearby training points exist

Now predict at temp=500 (way outside your data):
1. Similarity to all training points is low
2. Weighted average is unreliable
3. Uncertainty: large

This is exactly the right behavior — be confident near data, be uncertain far from data.

---

---

# Part 2 — Bayesian Optimization

---

## 2.1 What is Bayesian Optimization?

Bayesian Optimization (BO) is a strategy for finding the maximum (or minimum) of an expensive function using as few evaluations as possible.

"Expensive" means:
- Running a real chemical reactor experiment
- A 6-hour CFD simulation
- A drug synthesis that costs $10,000 per trial

You can't just grid search or run 10,000 random experiments. You need to be **smart** about which experiments to run next.

BO does this by combining two things:
1. **A GP** to model your function and track uncertainty
2. **An acquisition function** to decide where to sample next

---

## 2.2 The loop

Bayesian Optimization runs in a loop:

```
1. Build a GP from your current data
   → gives you predictions + uncertainty everywhere

2. Use acquisition function to find the most promising point
   → balances "go where prediction is high" vs "go where uncertainty is high"

3. Run the actual experiment at that point
   → get the real output

4. Add new data point to your dataset

5. Go back to step 1
```

Each iteration you run exactly one experiment (or a small batch), chosen intelligently. After 20-30 iterations you've usually found a very good optimum.

---

## 2.3 The acquisition function — the decision maker

The GP tells you what you know. The acquisition function tells you **what to do with that knowledge**.

The most important one to understand is **Expected Improvement (EI)**:

$$EI(x) = \mathbb{E}[\max(f(x) - f^*, 0)]$$

Where $f^*$ is the best yield you've seen so far.

In plain English: "how much improvement over my current best do I expect if I run an experiment at $x$?"

This naturally balances two competing goals:

**Exploitation** — go to points where predicted yield is high
> "I think temp=375 gives yield=0.87, which beats my current best of 0.85"

**Exploration** — go to points where uncertainty is high
> "I've never tried temp=450, I'm very uncertain there, it might be great"

A point with high predicted yield AND high uncertainty scores very high on EI. The algorithm will naturally explore uncertain regions that might be good, while also refining around known good regions.

---

## 2.4 Why not just go to the highest predicted point?

This is pure **exploitation** — always go where the GP mean is highest. The problem:

```
Your data:  temp 300-420 all explored
GP thinks:  temp=380 is best so far

If you always pick highest mean:
  → you keep running experiments near temp=380
  → you never discover that temp=480 is actually much better
  → you get stuck in a local optimum
```

Exploration prevents this. By occasionally sampling uncertain regions, you might discover better peaks you didn't know existed.

The acquisition function automatically handles this tradeoff — you don't need to decide manually.

---

## 2.5 Multi-task Bayesian Optimization (what your notebook does)

Standard BO optimizes one objective (yield).

Multi-task BO optimizes **multiple related objectives at once**, or uses data from related tasks to speed up optimization.

Example in chem eng:
- Task 1: optimize yield in Reactor A
- Task 2: optimize yield in Reactor B (similar design, slightly different)

Instead of running BO independently for each reactor, multi-task BO shares information between them. What it learns about Reactor A helps it make smarter decisions about Reactor B.

The key idea: if two tasks are similar (correlated), data from one is informative about the other. The GP is extended to model this correlation across tasks, not just across input space.

This is powerful when:
- You have expensive experiments across multiple related systems
- One task has more data than another (transfer learning)
- You want to find conditions that work well across all tasks simultaneously (robust optimization)

---

## 2.6 The connection back to kernels

Everything in GP and BO comes back to the kernel:

- The kernel defines what "similar inputs" means
- Similar inputs → correlated outputs → smooth GP predictions
- In multi-task BO, the kernel is extended to also measure similarity **between tasks**

A common choice is a product kernel:

$$K((x, t), (x', t')) = K_{\text{input}}(x, x') \times K_{\text{task}}(t, t')$$

Where:
- $K_{\text{input}}$ is the RBF kernel over your input features (temp, pressure, etc.)
- $K_{\text{task}}$ is a learned matrix that says how correlated Task 1 and Task 2 are

If the tasks are very similar, $K_{\text{task}}(1, 2)$ will be large → lots of information sharing.
If the tasks are unrelated, $K_{\text{task}}(1, 2)$ ≈ 0 → they learn independently.

---

## Summary so far

```
Kernel
  → measures similarity between data points
  → defines what smooth means

Gaussian Process
  → uses kernel to build a probabilistic model
  → gives predictions + uncertainty at every point
  → confident near data, uncertain far from data

Bayesian Optimization
  → uses GP to model an expensive function
  → uses acquisition function to decide where to sample next
  → balances exploitation (go where it's good) vs exploration (go where you're uncertain)

Multi-task BO
  → extends GP to model multiple related tasks
  → shares information across tasks via a task kernel
  → more efficient when tasks are correlated
```

---

## What to read next

- **Chapter 2** of *Gaussian Processes for Machine Learning* — Rasmussen & Williams (free at gaussianprocess.org) — the definitive textbook
- **Chapter 6** of *Pattern Recognition and Machine Learning* — Bishop — kernels and GPs from a Bayesian perspective
