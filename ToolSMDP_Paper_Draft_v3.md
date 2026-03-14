# ToolSMDP: Segment-Level Reinforcement Learning for LLM Tool Use via the Semi-Markov Decision Process Framework

*Anonymous Authors*

---

## Abstract

Teaching large language models (LLMs) to use external tools through reinforcement learning (RL) is an active area of research. Existing approaches treat the entire multi-tool trajectory as a single RL rollout, applying one outcome reward to the complete generation. This creates two problems: a credit assignment problem (when multiple tool calls contribute to success, which deserves the credit?) and a tool-use efficiency problem (models learn to over-rely on tools even for questions answerable from parametric knowledge, because trajectory-level rewards cannot distinguish necessary from unnecessary tool calls). We propose **ToolSMDP**, a segment-level RL framework grounded in the Options/Semi-Markov Decision Process (SMDP) theory (Sutton, Precup & Singh, 1999). Each generation segment—from context to the next tool invocation or end-of-sequence—constitutes an *option* in the SMDP sense. A learned value function provides per-segment credit assignment: intermediate segments are evaluated by whether they improved the critic's prediction of success (V(s') − V(s)), not by the final reward. This means a bad tool call receives negative advantage even within an overall successful trajectory, and an unnecessary tool call receives near-zero advantage because V was already high. We use Python code blocks as a universal tool interface requiring no supervised fine-tuning, and employ comment-preserving replacement where executed code is replaced by its output. Gradient masking becomes unnecessary by construction: each backward pass covers only model-generated tokens within one segment.

---

## 1. Introduction

### 1.1 Two Problems with Trajectory-Level Tool-Use RL

Current RL approaches to teaching LLMs to use tools—Search-R1 (Jin et al., 2025), ToRL (Li et al., 2025), ToolRL (Qian et al., 2025)—treat the entire multi-tool trajectory as a single RL episode with one outcome reward. This creates two distinct problems.

**The credit assignment problem.** Consider a model answering: *"What percentage of France's GDP does its military spending represent?"* This requires two tool calls: searching for military spending, then searching for GDP. Suppose the model executes a good first search (finding €50 billion in military spending), a bad second search (accidentally querying France's population instead of GDP), recovers using parametric knowledge, and produces the correct answer. In Search-R1's GRPO, the normalized advantage is a single scalar applied to every model-generated token. The good search query, the bad search query, the recovery reasoning, and the final answer all receive identical positive reinforcement. The model is told "do more of everything you did here"—including the wrong search.

With PPO and a token-level value function (also available in Search-R1), one might expect finer-grained credit. However, Generalized Advantage Estimation (GAE) computes each token's advantage as a discounted sum of all future TD residuals. In a 200-token trajectory where the bad search is at position 80, the negative signal is overwhelmed by approximately 120 subsequent positive contributions from recovery reasoning, making the net advantage at the bad search token positive despite the search itself being counterproductive.

**The tool-use efficiency problem.** Most post-training techniques for tool use—both prompt-based methods and RL approaches like Search-R1—incentivize models to over-rely on external tools, even for questions easily answerable from parametric knowledge. Consider: *"What is the GDP of the country that has California as a state?"* Every LLM above 1B parameters knows California is in the United States. Yet prompt-based methods (RAG, ReAct) instruct the model to always search for factual questions, and GRPO-based RL methods like Search-R1 positively reinforce any search call that appears in a successful trajectory—regardless of whether the search was necessary. Search-R1's training curves show search frequency increasing monotonically (their Figure 2d), with no mechanism to learn selective tool use.

This inefficiency is not merely a compute concern. Unnecessary tool calls introduce latency, increase the chance of retrieval errors, and lengthen the context window—all of which can degrade answer quality on questions the model could have answered directly.

### 1.2 Segment-Level Credit Assignment via SMDP

Our approach decomposes the trajectory into segments at tool-call boundaries. Each segment is an option in the classical RL sense (Sutton, Precup & Singh, 1999). The advantage for each segment is computed independently:

```
Intermediate segment:  A(segment_k) = V(s_{k+1}) - V(s_k)
Final segment:         A(segment_N) = R - V(s_N)
```

This formulation addresses both problems simultaneously.

**Credit assignment.** For the France example, this produces three independent signals. Segment 1 (good search): V(s₁) − V(s₀) = 0.65 − 0.35 = +0.30—the critic learned that military spending data substantially improves expected success. Segment 2 (bad search): V(s₂) − V(s₁) = 0.55 − 0.65 = −0.10—despite the trajectory succeeding (R=1), the critic detected that the context worsened after irrelevant population data was added. Segment 3 (recovery): R − V(s₂) = 1.0 − 0.55 = +0.45—the model salvaged a bad position, which is strongly reinforced.

Crucially, the reward R appears only in the final segment's advantage. Intermediate segment advantages depend entirely on the value function's assessment of state changes, not on the trajectory's outcome.

**Selective tool use.** For the California/GDP question where V(s₀) is already high (the critic knows the model can answer this directly), an unnecessary search produces near-zero advantage: V(s₁) − V(s₀) ≈ 0.75 − 0.70 = +0.05. Answering directly without tools produces stronger advantage: R − V(s₀) = 1.0 − 0.70 = +0.30. The model learns that direct answers on easy questions yield stronger reinforcement than unnecessary tool calls, because the value function implicitly encodes which questions the model can handle from parametric knowledge (high V(s₀)) versus which genuinely require tools (low V(s₀)).

### 1.3 Tool Interface Design: Python Code Blocks with Comment-Preserving Replacement

We use Python code blocks as a universal tool interface. Code-pretrained base models already generate markdown-style code fences with high probability, requiring no supervised fine-tuning on tool-call syntax. The "tool" is the Python interpreter itself—the model can write arithmetic, call a predefined `search()` function, or implement custom algorithms.

When a code block is executed, we extract any leading comment lines (the model's annotation of intent), execute the remaining code, and replace the entire code block with comments plus stdout:

```
Before: "I need France's GDP.\n```python\n# France GDP lookup\nresult = search('France GDP 2024')\nprint(result)\n```"

After:  "I need France's GDP.\n# France GDP lookup\nFrance's GDP in 2024 was approximately 3.05 trillion USD..."
```

The code vanishes; the comment and result remain as natural text. This produces coherent context for subsequent generation.

### 1.4 No Gradient Masking—By Construction

In Search-R1/ToRL, the complete trajectory contains both model-generated and environment-injected tokens, requiring gradient masking (25% performance degradation without it in Search-R1). In our formulation, each segment is a separate training episode where every token was generated by the model. Tool outputs appear only in the prompt of the next segment. The gradient masking problem is eliminated as a structural byproduct of the segment decomposition.

### 1.5 Contributions

1. **Segment-level credit assignment via SMDP.** Per-segment advantage computation through a learned value function. Good tool calls receive positive advantage, bad tool calls receive negative advantage, and unnecessary tool calls receive near-zero advantage—all from a single outcome reward.

2. **Analysis of why segment-level advantage outperforms token-level GAE.** GAE within a long trajectory smears credit temporally; segment-level advantage eliminates this by construction.

3. **Natural selective tool use.** The value function implicitly learns which questions require tools (low V(s₀)) versus which are answerable from parametric knowledge (high V(s₀)), creating pressure toward efficient tool use without explicit tool-use penalties.

4. **Python code blocks as universal tool interface with comment-preserving replacement.** No SFT required for tool-call syntax; coherent context for multi-turn tool use.

5. **Self-correcting tool-use learning.** Shared code syntax tokens appear in both positive and negative segments, causing gradients to cancel for syntax while content-specific tokens receive directional signal.

---

## 2. Related Work

### 2.1 RL for LLM Tool Use

| Property | Search-R1 | ToRL | ToolRL | StepTool | **Ours** |
|---|---|---|---|---|---|
| RL formulation | Single-rollout | Single-rollout | Single-rollout | Step-grained | **Multi-step SMDP** |
| Gradient masking | Required | Required | Required | Required | **Not needed** |
| Credit assignment | Trajectory-level | Trajectory-level | Trajectory-level | Per-step reward model | **Value function bootstrapping** |
| Tool-use efficiency | No selective pressure | No selective pressure | No selective pressure | No selective pressure | **V(s₀) provides natural selectivity** |
| Reward design | Outcome-only | Outcome + exec penalty | Multi-granularity | Step reward + outcome | **Outcome-only** |
| Tool types | Search only | Code interpreter | API functions | API functions | **Any (Python)** |
| Theoretical basis | Standard PPO/GRPO | Standard GRPO | Standard PPO | None | **Options/SMDP** |

Search-R1 (Jin et al., 2025) trains LLMs to call search engines using outcome-only RL on a single-rollout trajectory. ToRL (Li et al., 2025) applies a similar approach to code interpreter use, finding that intermediate code-execution penalties hurt performance. ToolRL (Qian et al., 2025) extends to API function calling. All treat the entire trajectory as a single RL episode.

### 2.2 The Options Framework and SMDPs

The Options framework (Sutton, Precup & Singh, 1999) provides mathematical foundations for temporal abstraction in RL. An option ⟨I, π, β⟩ consists of an initiation set, an intra-option policy, and a termination condition. Theorem 1 establishes that any MDP augmented with options constitutes an SMDP. The Option-Critic architecture (Bacon, Harb & Precup, 2017) extends this to end-to-end learning. Our work is the first to formally map LLM tool-use segments to the Options framework.

### 2.3 Credit Assignment in Delayed-Reward Settings

RUDDER (Arjona-Medina et al., 2019) decomposes delayed returns via contribution analysis. Potential-based reward shaping (Ng et al., 1999) proves that shaped rewards F(s,s') = γΦ(s') − Φ(s) preserve optimal policies. Our learned value function serves as an implicit potential function, providing equivalent credit assignment without manual design.

---

## 3. Methodology

### 3.1 Problem Formulation as a Semi-Markov Decision Process

**States.** A state s ∈ S is the accumulated textual context at a segment boundary. The initial state s₀ = q (the question plus system prompt). After segment k produces text t_k and a tool returns output o_k, the next state is s_{k+1} = s_k ⊕ comments_k ⊕ o_k, where comments_k are any leading comment lines from the code block and o_k is the tool's stdout. The executable code tokens are discarded.

**Options (= Generation Segments).** Each option ω_k = ⟨I_k, π_θ, β_k⟩ where I_k = {s_k} (initiates at current state), π_θ is the LLM's token-level policy (shared across all options), and β_k terminates when the model emits a closing code fence or EOS token.

**Reward.** A single reward R is given only at episode termination: R = EM(extract_answer(generation), gold_answer) ∈ {0, 1}. No intermediate rewards, no tool-execution rewards, no format rewards.

**Transitions.** When option ω_k terminates via code block detection: the code is executed in a sandboxed Python interpreter, stdout is captured, and the new state is constructed via comment-preserving replacement. The model never sees the executed code again—only the comments and output persist in context.

**Transition determinism and partial observability.** In standard LLM generation, state transitions are deterministic: appending a token to the context always produces the same next state. Transition probability is identically 1, with all stochasticity residing in the policy (which token to sample). Our comment-preserving replacement design introduces partial observability: executable code is erased from the state after tool execution, replaced by the tool's stdout and any leading comments. The next state no longer contains the code that caused the tool invocation—only its intent annotation and result. Different code paths producing different stdout yield different next states from the same starting state, making the transition at the SMDP level genuinely non-deterministic. Formally, this makes our formulation a partially observed SMDP (POSMDP). However, this does not affect the validity of our training procedure for three reasons. First, the value function V(s) remains well-defined over observable states—the critic evaluates the post-replacement context, which contains the tool output (the information that predicts future success) even though the code syntax is lost. Second, the segment-level advantage V(s_{k+1}) − V(s_k) remains meaningful as a measure of whether the observable state improved after tool execution. Third, the policy gradient is unaffected: during the backward pass for each segment, the policy sees its own complete generation including the code and receives gradients through all generated tokens. The partial observability affects the critic's input, not the policy's training signal.

### 3.2 SMDP Bellman Equation and Convergence

By Theorem 1 of Sutton, Precup & Singh (1999), our formulation satisfies the conditions for an SMDP: states are well-defined textual contexts at segment boundaries, actions are options (generation segments), and transition dynamics are determined by the LLM's generation and tool execution. The SMDP Bellman equation for the value of state s under policy μ is:

```
V^μ(s) = Σ_ω μ(s, ω) [ r^ω_s + Σ_{s'} p^ω_{ss'} V^μ(s') ]
```

With γ = 1 (episodic task, no discount), this simplifies to standard undiscounted Bellman equations. We note that the SMDP framework provides structural insight and a principled decomposition rather than neural convergence guarantees, since our value function uses neural network approximation on a combinatorially large state space.

### 3.3 Value Function and Segment-Level Advantage

**Critic architecture.** We attach a value head V_φ (single linear layer) to the LLM's last hidden state. For state s_k, V_φ(s_k) = Linear(h_last(s_k)), where h_last is the LLM's last-layer hidden state at the final token position of the prompt s_k. The critic shares the LLM backbone with the policy—no separate model is needed. With ~3,584 trainable parameters, the critic head adds negligible memory and compute.

**Segment-level advantage.** For a trajectory with N segments:

```
Final segment (EOS):         A(seg_N) = R - V_φ(s_N)
Intermediate segment (tool):  A(seg_k) = V_φ(s_{k+1}) - V_φ(s_k)
```

The advantage for intermediate segments does not involve the reward R. It depends entirely on the value function's assessment of how the context changed after tool execution.

**Grounding in Options theory.** In the Options framework, an option is treated as an atomic action at the SMDP level. The advantage A(ω_k) = V(s_{k+1}) − V(s_k) is the advantage of the option as a whole. All tokens within a segment share this option-level advantage, consistent with the SMDP policy gradient: ∇ log π(option | state) × A(option) expands to ∇ Σ_t log π(token_t | context_t) × A(option), because the probability of the option is the product of its constituent token probabilities.

### 3.4 Why Segment-Level Advantage Outperforms Token-Level GAE

PPO with a token-level value function computes advantages via GAE. With terminal-only reward and γ = 1, GAE computes:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ... + (γλ)^{T-t} δ_T
```

With λ = 0.95, a bad tool call producing a negative δ at position t is followed by ~100 tokens of recovery reasoning, each contributing small positive δs. These accumulate to approximately +0.3, overwhelming the single negative δ of −0.10. The net advantage at the bad tool call becomes positive.

In our formulation, the advantage is simply V(s_{k+1}) − V(s_k) = −0.10. No GAE sum, no temporal smearing. The recovery has its own independent advantage (+0.45) in a separate segment.

**This is the core mechanical advantage: GAE within a trajectory smears credit across time, while segment-level advantage eliminates this smearing by construction.**

Additionally, Search-R1's token-level value function operates on sequences containing both model-generated and environment-injected tokens, a harder learning problem than evaluating clean prompts at segment boundaries.

### 3.5 How the Value Function Enables Selective Tool Use

The value function V(s₀) implicitly encodes the difficulty of a question relative to the model's parametric knowledge. For questions the model can answer directly, V(s₀) is high (the critic expects success without tools). For questions requiring external information, V(s₀) is low.

This creates a natural gradient toward selective tool use:

| Scenario | Tool-call advantage | Direct-answer advantage | Model learns |
|---|---|---|---|
| Easy question, unnecessary tool | V(s₁)−V(s₀) ≈ +0.05 (small) | R−V(s₀) ≈ +0.30 (large) | Direct answer is more efficient |
| Hard question, necessary tool | V(s₁)−V(s₀) ≈ +0.45 (large) | Would get R=0 without tool | Tool use is essential |
| Hard question, bad tool call | V(s₁)−V(s₀) ≈ −0.15 (negative) | — | Penalize bad tool call |

No explicit tool-use penalty is needed. The value function's state assessments naturally create stronger reinforcement for tool use when it is necessary and weaker reinforcement when it is redundant.

### 3.6 Self-Correcting Tool-Use Learning

A natural concern is whether penalizing a bad-tool-call segment might cause the model to avoid generating code entirely. Three self-correcting mechanisms prevent this.

**Shared syntax cancellation.** Code syntax tokens (function calls, print statements, code fences) appear in both positively and negatively advantaged segments across the training batch. Over many updates, gradient contributions for shared syntax approximately cancel, while content-specific tokens (search queries, variable values) receive consistent directional signal. The model learns *what* to compute, not *whether* to compute.

**Magnitude asymmetry.** A mildly bad tool call receives small negative advantage (−0.10). Completely avoiding tools on a hard question receives larger negative advantage (R − V(s₀) = 0 − 0.35 = −0.35). Using tools badly is preferable to not using tools at all on questions requiring external information.

**Strong pretraining prior.** Code-pretrained models have deeply encoded priors for generating syntactically valid code. RL fine-tuning adjusts *what* the model writes in code blocks, not *whether* it writes them.

### 3.7 Why PPO, Not GRPO

GRPO (Shao et al., 2024) computes advantages at the trajectory level: every segment in a trajectory receives the same advantage. GRPO is structurally incompatible with segment-level credit assignment and cannot provide the selective tool-use pressure described above (it cannot distinguish V(s₀) for easy vs. hard questions). PPO with a learned value function is essential for our approach.

### 3.8 Training Data Curation

We pre-filter the training data by evaluating the base model without tool access, retaining primarily questions that the base model answers incorrectly (Tier 1, ~70% of training batches). A minority of solvable questions (Tier 2, ~30%) is retained to prevent tool over-reliance and provide stable reward signal during early training. We use a simple curriculum: early batches weight toward single-tool-call questions, gradually shifting toward multi-tool questions. A system prompt informs the model of available tools: *"You can write Python code in code blocks. Code will be executed and you will see the output. Available functions: search(query). Always begin code blocks with a comment explaining what you are computing."*

---

## 4. Reward Design: Why Outcome-Only Works

We use a single binary reward: R = EM(extracted_answer, gold_answer) ∈ {0, 1}. No per-segment rewards, no tool-execution rewards, no length penalties.

**Why not reward tool execution success?** ToRL tested a code-executability penalty (−0.5). Performance decreased: models learned trivially simple code to avoid the penalty. Our value function handles this correctly—bad code → bad tool output → low V(s_{k+1}) → negative segment advantage—without intermediate reward engineering.

**Why not reward tool-use efficiency explicitly?** One could add a penalty for unnecessary tool calls. But our value function provides this implicitly: V(s₀) is already high for easy questions, so unnecessary tool calls produce near-zero advantage while direct answers produce strong positive advantage. No explicit efficiency penalty is needed.

**The value function as implicit potential function.** Our advantage A(seg_k) = V(s_{k+1}) − V(s_k) has the same form as potential-based reward shaping F(s,s') = Φ(s') − Φ(s) (Ng et al., 1999), with the guarantee that a learned Φ captures task-specific value that no hand-crafted potential could match.

---

## 5. Analysis of Terminal Cases

| Case | Trajectory | R | Seg 1 Adv | Seg 2+ Adv | What Model Learns |
|---|---|---|---|---|---|
| **A: Direct answer (easy Q)** | s₀ → answer EOS | 1.0 | R−V(s₀) = +0.30 | — | Direct answers on easy questions |
| **B: Necessary tool** | s₀ → good code → s₁ → answer | 1.0 | V(s₁)−V(s₀) = +0.55 | R−V(s₁) = +0.05 | Tools for hard questions |
| **C: Unnecessary tool** | s₀ → redundant search → s₁ → answer | 1.0 | V(s₁)−V(s₀) = +0.05 | R−V(s₁) = +0.25 | Weak signal for unnecessary tools |
| **D: Bad tool call** | s₀ → bad code → s₁ → wrong | 0.0 | V(s₁)−V(s₀) = −0.90 | R−V(s₁) = +0.50 | Bad calls strongly penalized |
| **E: Error → recovery** | s₀ → typo → s₁ → fix → s₂ → ans | 1.0 | −0.20 | +0.75 | Recovery is very valuable |
| **F: Tool avoidance (hard Q)** | s₀ → wrong answer EOS | 0.0 | R−V(s₀) = −0.35 | — | Avoiding tools on hard Q penalized |

**Key comparisons.** Case A vs Case C: direct answers on easy questions receive +0.30 advantage while unnecessary tool calls receive only +0.05—creating pressure toward efficiency. Case B vs Case F: using tools on hard questions receives +0.55 while avoiding tools receives −0.35—creating pressure toward tool use when needed. Case D vs Case E: bad tool calls are penalized (−0.90 or −0.20) but recovery is strongly rewarded (+0.75)—the model learns to recover from errors rather than being permanently punished for them. None of these distinctions are possible with trajectory-level GRPO.

---

## 6. Comparison with Search-R1 and ToRL

### 6.1 Structural Differences

| Dimension | Search-R1 / ToRL | ToolSMDP (Ours) |
|---|---|---|
| RL Episode | Full trajectory | One segment per episode |
| Gradient masking | Required (25% perf drop without) | Not needed (by construction) |
| Credit assignment | Same advantage for all segments | Independent advantage per segment |
| Bad tool call in successful trajectory | Positive advantage | Negative advantage |
| Unnecessary tool call | Positive advantage (same as necessary) | Near-zero advantage (V already high) |
| GAE smearing | Future positive δs dilute bad signal | No smearing—clean V(s')−V(s) |
| Tool-use selectivity | No mechanism (tools always reinforced) | V(s₀) encodes question difficulty |
| Tool call syntax | Custom tokens | Python code blocks (no SFT) |
| RL algorithm | PPO or GRPO | PPO (critic essential) |
| Theoretical basis | Standard PPO/GRPO | SMDP + Options framework |

### 6.2 Decomposed Skills

Segment-level structure decomposes tool use into independently trained skills. Segment 1 trains: "Should I invoke a tool and with what arguments?" Segment 2 trains: "Given the tool's result, how should I continue?" Skill 2 leverages pretraining directly—tool output appears as natural text in the prompt.

### 6.3 Predicted Advantage Distribution

A key empirical prediction: classifying tool-call segments by output relevance, the average advantage should show clear separation in ToolSMDP (positive for relevant, negative for irrelevant) and weak separation in Search-R1's PPO (both near-positive due to GAE smearing). Additionally, classifying by question difficulty, unnecessary tool calls on easy questions should show near-zero advantage in ToolSMDP but positive advantage in Search-R1.

---

## 7. Experiments

### 7.1 Training Configuration

**Base model:** Qwen2.5-7B-Base (used by both Search-R1 and ToRL). We additionally report results at 3B scale to demonstrate robustness.

**Framework:** OpenRLHF with PPO and segment-level value function. Search-R1 GRPO baseline reimplemented in the same framework for fair comparison.

**Training data:** Merged NQ and HotpotQA training splits for search experiments (~170K questions), GSM8K and MATH for math experiments (~15K questions), FinQA for multi-tool experiments (~6.2K questions).

**Hyperparameters:** Learning rate 1.5e-6 (policy), 5e-6 (critic head), KL coefficient 0 (following R1-Searcher for base models), PPO clip ε = 0.2, temperature 1.0, max 5 segments per question.

### 7.2 Evaluation Strategy

We evaluate on benchmarks grouped by the number of tool calls they naturally require, testing our hypothesis that gains increase with credit assignment complexity:

**Multi-hop QA (2-4 tool calls, primary benchmarks):** HotpotQA, Musique, 2WikiMultiHopQA. These are our main results—the benchmarks where segment-level credit should provide the largest advantage.

**Math reasoning (1-3 tool calls, competitiveness):** GSM8K, MATH. We expect comparable performance to ToRL on single-step problems, with advantages on multi-step computation.

**Multi-tool routing (2-3 heterogeneous tool calls, novel):** FinQA. Requires both retrieval and calculation—no prior RL baseline exists.

**Single-hop QA (0-1 tool calls, sanity check):** NQ, TriviaQA. Reported in text only: we expect comparable performance since credit assignment advantage is minimal with a single tool call.

**Difficulty-bucketed evaluation.** For each benchmark, we partition questions by the number of tool calls used in base-model rollouts (pre-training analysis) and report EM per bucket:

**Table 1: Main Results — Multi-Hop QA**

| Method | HotpotQA EM | HotpotQA F1 | 2Wiki EM | Musique EM | Avg EM |
|---|---|---|---|---|---|
| Qwen2.5-7B-Base (no tools) | — | — | — | — | — |
| Search-R1 (reproduced, GRPO) | — | — | — | — | — |
| **ToolSMDP (Ours)** | **—** | **—** | **—** | **—** | **—** |

**Table 2: Main Results — Math Reasoning**

| Method | GSM8K Acc | MATH Acc |
|---|---|---|
| Qwen2.5-7B-Base (no tools) | — | — |
| ToRL (reported) | — | — |
| **ToolSMDP (Ours)** | **—** | **—** |

**Table 3: Multi-Tool Routing (Novel)**

| Method | FinQA EM | FinQA F1 |
|---|---|---|
| Qwen2.5-7B-Base (no tools) | — | — |
| **ToolSMDP (Ours)** | **—** | **—** |

**Table 4: Gains by Number of Tool Calls (Key Result)**

| Method | 1 tool call | 2 tool calls | 3+ tool calls |
|---|---|---|---|
| Search-R1 (GRPO) | — | — | — |
| ToolSMDP + GRPO | — | — | — |
| **ToolSMDP + PPO (Ours)** | **—** | **—** | **—** |

*We predict gains increase with the number of tool calls, directly validating the segment-level credit assignment mechanism.*

### 7.3 Key Ablation: Isolating the Credit Assignment Mechanism

| Variant | Episode Structure | Credit Assignment | Gradient Mask | HotpotQA | Musique | GSM8K |
|---|---|---|---|---|---|---|
| Search-R1 (reproduced) | Single rollout | Trajectory (GRPO) | Yes | — | — | — |
| ToolSMDP + GRPO | Segment | Trajectory (GRPO) | No | — | — | — |
| **ToolSMDP + PPO (Ours)** | **Segment** | **Segment (V(s) critic)** | **No** | **—** | **—** | **—** |

If PPO > GRPO ≈ Search-R1: the value function is the driver. If PPO > GRPO > Search-R1: both segment structure and credit assignment contribute.

### 7.4 Additional Ablations

**Table 5: Reward Design Ablation**

| Reward Variant | HotpotQA EM | GSM8K Acc |
|---|---|---|
| Outcome-only (default) | — | — |
| Outcome + execution penalty | — | — |
| Outcome + length penalty | — | — |

**Table 6: Value Function Ablation**

| Value Target | HotpotQA EM | GSM8K Acc |
|---|---|---|
| Bootstrap V(s_{k+1}) (default) | — | — |
| Monte Carlo (R for all segments) | — | — |

### 7.5 Diagnostic Analyses

1. **Value function calibration** — scatter plots of V(s_k) vs actual episode R.

2. **Advantage distributions per segment** — average advantage for relevant vs irrelevant tool outputs. Clear separation validates the credit assignment mechanism.

3. **Tool call frequency and selectivity over training** — fraction of trajectories using tools, broken down by question difficulty (Tier 1 vs Tier 2). We predict tool frequency increases for hard questions and decreases (or stays low) for easy questions—demonstrating learned selectivity.

4. **Error recovery rate** — how often the model recovers from bad tool calls in subsequent segments.

5. **Training curve comparison** — EM over training steps for ToolSMDP vs Search-R1 vs GRPO variant. We predict an initial phase of comparable performance (critic calibrating) followed by a crossover.

6. **Unnecessary tool call rate** — fraction of easy questions (Tier 2) where the model uses tools, compared across methods. We predict ToolSMDP produces fewer unnecessary tool calls than Search-R1.

---

## 8. Expected Results and Hypotheses

**H1:** Segment-level credit assignment will show the largest gains on multi-hop tasks (HotpotQA, Musique) where multiple tool calls are needed. Gains should increase with the number of tool calls per question (Table 4).

**H2:** On single-tool tasks (GSM8K, NQ), performance will be comparable to baselines.

**H3:** Training curves will show an initial phase where ToolSMDP performs comparably to Search-R1 (critic calibrating), followed by a crossover.

**H4:** On FinQA (multi-tool), our approach will outperform baselines, demonstrating implicit tool routing.

**H5:** ToolSMDP-trained models will make fewer unnecessary tool calls on easy questions than Search-R1-trained models, demonstrating learned selectivity.

---

## 9. Conclusion

We presented ToolSMDP, a segment-level RL framework for LLM tool use grounded in the Options/SMDP theory. The core contribution is a credit assignment mechanism that provides independent learning signals for each tool-call segment: good tool calls receive positive advantage, bad tool calls receive negative advantage, unnecessary tool calls receive near-zero advantage, and recovery is strongly reinforced—all from a single outcome reward. The value function simultaneously enables selective tool use by implicitly encoding question difficulty: models learn to use tools when parametric knowledge is insufficient and answer directly when it suffices. This is achieved through segment-level evaluation that avoids GAE temporal smearing, uses Python code blocks as a universal tool interface (no SFT), employs comment-preserving replacement for coherent context, and eliminates gradient masking by construction.

---

## References

- Arjona-Medina, J.A., et al. (2019). RUDDER: Return Decomposition for Delayed Rewards. NeurIPS.
- Bacon, P.-L., Harb, J., & Precup, D. (2017). The Option-Critic Architecture. AAAI.
- Guo, D., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL. arXiv:2501.12948.
- Jin, B., et al. (2025). Search-R1: Training LLMs to Reason and Leverage Search Engines with RL. COLM 2025.
- Li, X., Zou, H., & Liu, P. (2025). ToRL: Scaling Tool-Integrated RL. arXiv:2503.23383.
- Ng, A.Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations. ICML.
- Qian, C., et al. (2025). ToolRL: Reward is All Tool Learning Needs. NeurIPS 2025.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning. arXiv:2402.03300.
- Sutton, R.S., Precup, D., & Singh, S. (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL. Artificial Intelligence 112, 181–211.

---

## Appendix A: Context Format Variant

Our default design (comment-preserving replacement) erases executable code from the state after tool execution, retaining only comments and stdout. An alternative design (append) retains the full code block alongside the tool output, producing a fully observed SMDP with deterministic transitions. The segment-level credit assignment mechanism is identical in both designs. If compute budget permits, comparing replacement against append would isolate whether the information lost through code erasure affects the critic's ability to predict future success. We expect minimal difference, since the tool's output is far more predictive of future success than the code syntax that produced it.
