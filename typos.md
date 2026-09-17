# Paper Typos, Omissions, and Reproduction Assumptions

This document compiles all typographical errors, arithmetic discrepancies, missing parameters, and ambiguities identified in the original paper:

> **Reference:** Sandoval-Arrayga, C. J., Palacios-Ramirez, G., & Ramos-Corchado, F. F. (2024). *Temporal heterogeneity in cognitive architectures*. **Cognitive Systems Research**, 88, 101265. [DOI: 10.1016/j.cogsys.2024.101265](https://doi.org/10.1016/j.cogsys.2024.101265)

It also details the technical rationale and assumptions implemented in [`configs/config.yaml`](./configs/config.yaml) to ensure a stable, faithful reproduction.

---

## 📋 Summary Table

| # | Item / Parameter | Paper Statement | Corrected / Assumed Value | Config Key | Impact if Uncorrected |
|---|---|---|---|---|---|
| **1** | **Thinking Energy Cost** | `0.5% / 6 = 0.833%` (Sec 5.2) | `0.08333%` (Table 2: `.0833%`) | `agent.thinking_cost_per_layer: 0.0833` | Total thinking cost becomes 5.0%/step; agents die in ~18 steps. |
| **2** | **Food Energy Gain** | Promised in Sec 5.2 ("described in Table 2") | Completely absent. Assumed `25.0%` | `agent.food_energy_gain: 25.0` | Agents cannot replenish energy; 100% starvation rate. |
| **3** | **Hebbian Learning Rate ($\eta$)** | Omitted everywhere in text & tables | Assumed `1e-3` ($0.001$) | `cognitive_architecture.learning_rate: 1e-3` | Sanger GHA weights either diverge or fail to learn during lifespan. |
| **4** | **Simulation Horizon** | "45,000 runs" (Sec 5.4) | 45,000 **generations** of a single evolutionary trajectory | `simulation.generations: 100`<br>*(45,000 for full replication)* | 45,000 full multi-generation runs would be computationally intractable. |
| **5** | **Agent Lifetime Limit** | None; live while food/energy > 0 (Sec 5.4) | Capped at 10,000 steps | `agent.max_agent_steps: 10_000` | Cyclical foraging causes infinite loops in generation 1. |
| **6** | **Network Architecture** | "Two-layer neural network" (Fig 3 caption) | Single-layer linear mapping $y = Wx$ | `src/components/layer.py` | Adding hidden layers violates Sanger (1989) PCA convergence. |
| **7** | **Textual Slips** | "amount of food remained positive" (Sec 5.4) | Refers to agent internal **energy** | N/A | Semantic confusion between environment items and internal states. |
| **8** | **Fitness Plotting Metric** | "only the average of the top 10 agents with the highest fitness during that run was plotted" (Figs. 6 & 7 captions) | Monotonically non-decreasing cumulative maximum ($\text{cummax}$) of top-10 fitness | `src/simulation/visualization.py` | Direct raw plotting exhibits strong inter-generational variance, obscuring the step plateaus seen in Figures 6 & 7. |

---

## Detailed Analysis & Reproduction Decisions

### 1. Thinking Energy Cost: Arithmetic Typo (Section 5.2 vs Table 2)

* **Paper Location:** Section 5.2 (*Virtual Agents - Interoception*), paragraph before Table 2:
  > *"As there are a total of six perceptual layers and motor layers (see Fig. 3), the energy expended when each of these layers is executed equals 0.5%/6 = 0.833%. We term the process of executing each of these layers 'thinking'."*
* **The Discrepancy:**
  The calculation $\frac{0.5\%}{6}$ is mathematically equal to $\approx 0.08333\%$, **not** $0.833\%$. The authors shifted the decimal place by one order of magnitude in the text. However, in **Table 2**, the value is correctly listed as:
  > *"Energy consumption due to thinking .0833%"*
* **Impact:**
  If the typo value of $0.833\%$ were applied to each active layer, executing all 6 layers simultaneously would consume $6 \times 0.833\% = 5.0\%$ in thinking cost alone. Added to the movement cost of $0.5\%$, the total cost would reach $5.5\%$ per step, killing an agent (starting with $100\%$ energy) in fewer than 19 steps.
* **Our Implementation Decision:**
  In [`configs/config.yaml`](./configs/config.yaml):
  ```yaml
  agent:
    thinking_cost_per_layer: 0.0833 # 0.5% / 6 = 0.08333% (matches Table 2)
  ```
  This preserves the authors' stated design goal: a maximum possible step cost of $1.0\%$ ($0.5\%$ movement + $6 \times 0.08333\% \approx 0.5\%$ thinking).

---

### 2. Missing Food Energy Replenishment (Section 5.2 vs Table 2)

* **Paper Location:** Section 5.2 (*Virtual Agents - Interoception*), paragraph after Table 3:
  > *"For this study, we opted to create an energy level bar that depends on food to generate more energy. On the other hand, as the agent moves or thinks, it consumes energy. How energy is produced or consumed is described in Table 2."*
* **The Discrepancy:**
  Table 2 lists only:
  - *Total amount of energy:* 100%
  - *Energy consumption due to movement:* 0.5%
  - *Energy consumption due to thinking:* .0833%
  
  **How energy is produced (the energy gain from consuming food) is completely missing from Table 2 and is never quantified anywhere in the text.**
* **Impact:**
  Without a value for food replenishment, an agent cannot restore energy, rendering foraging useless and leading to guaranteed death in 100–200 steps.
* **Our Implementation Decision:**
  In [`configs/config.yaml`](./configs/config.yaml):
  ```yaml
  agent:
    food_energy_gain: 25.0 # Assumed +25.0% energy per food element
    max_energy: 100.0      # Energy cap
  ```
  A value of $+25.0\%$ provides balanced homeostasis: given a grid with $23\%$ food density, an agent that encounters food every 25–40 steps survives indefinitely if it navigates effectively, allowing evolutionary differences between architectures to emerge.

---

### 3. Omission of Sanger's Learning Rate $\eta$ (Section 5.2, Section 5.3, Section 9.2)

* **Paper Location:** Section 5.2 (*Ontogeny*), Section 5.3 (*Evolution*), and Section 9.2 (*Layers*).
* **The Discrepancy:**
  The paper specifies that each layer adapts online using Sanger's Generalized Hebbian Algorithm (GHA):
  $$\Delta W_{ij} = \eta \cdot y_i \left( x_j - \sum_{k=1}^i y_k W_{kj} \right)$$
  However, the learning rate parameter $\eta$ is **never reported** in the paper.
* **Impact:**
  Sanger's rule is sensitive to $\eta$:
  - If $\eta$ is too large ($\eta \ge 0.05$), weights rapidly explode.
  - If $\eta$ is too small ($\eta \le 10^{-6}$), ontogenetic adaptation during an agent's lifetime is negligible.
* **Our Implementation Decision:**
  In [`configs/config.yaml`](./configs/config.yaml):
  ```yaml
  cognitive_architecture:
    weight_init: 1e-4    # Initial weights in U(-1e-4, 1e-4) (Sec 5.3)
    learning_rate: 1e-3  # Assumed eta = 0.001 (1e-3)
  ```
  With weight initialization on the order of $10^{-4}$, $\eta = 10^{-3}$ ensures stable convergence toward the principal components of the inputs without numerical overflow, complemented by gradient clipping in [`src/components/layer.py`](./src/components/layer.py).

---

### 4. Evolutionary Scale: "45,000 Runs" vs Generations (Section 5.4 vs Figures 6 & 7)

* **Paper Location:** Section 5.4 (*Execution*), first paragraph:
  > *"For the execution of the model, 45,000 runs of the evolutionary process were conducted for each of the 6 models..."*
* **The Discrepancy:**
  The word *"runs"* is a mistranslation of the Spanish *"corridas"* (referring to cycles/generations of the genetic loop).
  - The x-axes of **Figure 6** and **Figure 7** are explicitly labeled **Generation**, spanning 0 to 45,000.
  - The figure captions state: *"only the average of the top 10 agents with the highest fitness during that run was plotted"*.
  - Section 6.1 mentions *"by the end of the 30000 evolution steps"*, which corresponds to generation 30,000.
* **Impact:**
  Interpreting 45,000 as independent runs would mean running $45,000 \times 45,000 \times 100$ agent lifetimes, which is infeasible and not what the authors performed.
* **Our Implementation Decision:**
  In [`configs/config.yaml`](./configs/config.yaml):
  ```yaml
  simulation:
    generations: 100 # Configurable: 45,000 for full paper reproduction, 100 for dev/testing
    num_runs: 1      # A single evolutionary trajectory per condition, matching paper figures
  ```

---

### 5. Maximum Agent Lifetime Safeguard (Section 5.4)

* **Paper Location:** Section 5.4 (*Execution*), second paragraph:
  > *"they were allowed to live as long as the amount of food they had remained positive"*
* **The Discrepancy:**
  In theory, if an agent learns a stable path connecting food items, its energy will never drop to 0, causing the simulation to hang in generation 1 forever.
* **Our Implementation Decision:**
  In [`configs/config.yaml`](./configs/config.yaml):
  ```yaml
  agent:
    max_agent_steps: 10_000 # Upper bound safeguard
  ```
  Setting `max_agent_steps: 10_000` prevents infinite loops while giving agents more than enough time to collect 40–50 coins (the asymptotic maximum observed in Figures 6 and 7).

---

### 6. Architectural Phrasing: "Two-Layer Neural Network" vs Linear Hebbian Layer

* **Paper Location:** Figure 3 caption:
  > *"Within each layer, there is a two-layer neural network utilizing the Hebbian learning algorithm as proposed by Sanger (1989)."*
* **The Discrepancy:**
  Sanger's Generalized Hebbian Algorithm (Sanger, 1989) is mathematically proven for a **single-layer linear feedforward network** ($y = Wx$, without hidden units or nonlinear activation functions). Calling it a "two-layer network" counts the input buffer and output buffer as layers.
* **Our Implementation Decision:**
  Implemented in [`src/components/layer.py`](./src/components/layer.py) as a direct matrix multiplication $y = Wx$ with Sanger's update rule $\Delta W = \eta y (x^T - y^T W)$, consistent with Sanger (1989) and Table 4.

---

### 7. Textual and Linguistic Errata

1. **"Food" instead of "Energy" (Section 5.4):**
   - Text: *"they were allowed to live as long as the amount of food they had remained positive"*.
   - Intended: Food is an environmental element; the agent's internal state that depletes and must remain positive is **energy**.
2. **Double Parenthesis / Redundant Referencing (Section 6.1):**
   - Text: *"and the last 3 using long-term memory (Figure (Fig. 7)."*
3. **Table 2 reference for energy production (Section 5.2):**
   - Text states that energy production is detailed in Table 2, but Table 2 contains no information on energy production.

---

### 8. Fitness Plotting Metric: Raw Generational Fitness vs. Cumulative Maximum (Figures 6 & 7)

* **Paper Location:** Section 5.4 (*Execution*), Section 6.1 (*Fitness*), and captions of Figures 6 & 7:
  > *"In both cases, only the average of the top 10 agents with the highest fitness during that run was plotted."*
* **The Discrepancy:**
  In evolutionary simulations with stochastic world generation, discrete agent foraging, and Lamarckian mutations, the generational mean fitness of the top 10 agents naturally fluctuates from generation to generation (producing a noisy, jagged curve).
  However, visual inspection of the actual plots in **Figure 6** and **Figure 7** reveals that every curve is **monotonically non-decreasing** (a step function with plateaus that only jump upward when a superior generation is discovered).
  The authors actually plotted the **cumulative running maximum** (the "best-so-far" top-10 fitness):
  $$F_{\text{plot}}(g) = \max_{1 \le i \le g} \bar{F}_{\text{top10}}(i)$$
* **Impact:**
  Plotting only raw generational fitness produces a high-variance graph resembling an electrocardiogram or stock chart, obscuring the characteristic evolutionary breakthroughs and step-wise adaptation plateaus reported in the paper.
* **Our Implementation Decision:**
  In [`src/simulation/visualization.py`](./src/simulation/visualization.py):
  1. We generate **both** visualization modes:
     - **Cumulative Maximum (`*_cummax.png` and default `figure_*.png`):** Strictly reproduces the step-function trajectories and plateaus of Figures 6 and 7.
     - **Raw Generational Fitness (`*_raw.png`):** Shows the true per-generation population variance and evolutionary exploration noise.
  2. We incorporate **black-and-white print-friendly visual aids**:
     - **Heterogeneity ($a=3$):** Solid line (`-`, dark tone `#08519c`, luminance-calibrated for B&W legibility).
     - **Homogeneity ($a=1$):** Dashed line (`--`, `dashes=(6, 3)`, warm amber `#d95f02`).
     - **Variance Bands:** Distinct hatch pattern (`//`) for homogeneity vs smooth alpha for heterogeneity.
     - **Legends:** Explicitly annotate the line style (`solid` vs `dashed`) so graphs are immediately distinguishable in monochrome or greyscale printouts.
