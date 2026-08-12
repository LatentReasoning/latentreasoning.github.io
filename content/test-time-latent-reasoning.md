---
title: "Reasoning as Latent-Space Optimization"
description: "A great latent space is one where complex problems yield to simple gradient ascent."
date: 2026-08-07
tags: [reasoning, test-time compute, latent space, interpretability]
bibliography: test-time-latent-reasoning.bib
authors:
  - name: "Hengli Li$^{1,2}$$^*$"
  - name: "Zilong Zheng$^1$$^✉$"
  - name: "Chi Zhang$^{1,2}$"
  - name: "Song-Chun Zhu$^{1,2}$"
  - name: "Ying Nian Wu$^3$"
affiliations:
  - "$^1$ NLCo Lab, Beijing Institute for General Artificial Intelligence"
  - "$^2$ School of Artificial Intelligence for Science, Peking University"
  - "$^3$ University of California, Los Angeles"
  - name: "$^*$ Core contributor"
    new_line: true
  - "$^✉$ Correspondence to zlzheng@bigai.ai"
citation_style: apa
---

Suppose a language model answers a math problem incorrectly. We could ask it again, sample a hundred solutions, grow a search tree, or update its weights. All of these responses spend more computation, but they spend it in different places.

There is another option: leave the model's weights untouched and optimize the _internal states of this one problem_. The model does not merely try another sentence. It adjusts the continuous vectors from which a sentence will be generated.

This is **test-time latent reasoning**. The phrase names a small but rapidly developing family of methods rather than a single algorithm, yet the family shares one commitment, and it gives this article its title: treat reasoning itself as optimization in latent space. Instance-specific hidden states become decision variables at inference time. A reward—perhaps correctness, model confidence, or image quality—defines a landscape over those states and supplies a direction of ascent. When the optimization ends, the states disappear; the next problem starts from the original frozen model.

<figure class="l-page">
  <img src="/images/test-time-latent-reasoning-figure-1.png" alt="Comparison of weight adaptation, token-space search, and test-time latent reasoning">
  <figcaption><strong>Figure 1.</strong> Three axes of test-time compute. Repeated sampling and tree search explore discrete trajectories; weight adaptation changes the model across or within instances; test-time latent reasoning optimizes temporary continuous states while keeping the model parameters fixed.</figcaption>
</figure>

## The interface is the bottleneck

Chain-of-thought (CoT) turned language into a computational scratchpad [@wei2022chain]. This is a remarkably useful interface: reasoning steps can be inspected, edited, verified, and fed back to the model. It is also restrictive. At every step, a high-dimensional hidden state must pass through a vocabulary-sized distribution and collapse to a discrete token. Once the token is sampled, most information in that distribution is gone.

The limitation matters most when a model is uncertain between several useful continuations. A latent vector can, in principle, carry aspects of many alternatives at once. A token must commit to one. Natural language also spends capacity on grammar, connective tissue, and exposition—properties valuable for communication but not necessarily for computation.

This observation has produced several responses.

- **Search over language.** Self-consistency samples multiple complete CoTs and votes on the final answer [@wang2023selfconsistency]. Tree of Thoughts scores and expands partial textual states [@yao2023tree]. The representation remains readable, but branching can be expensive.

- **Learn to reason continuously.** Implicit-CoT methods gradually hide textual steps during training [@deng2024implicit]. Coconut feeds the last hidden state back as the next input embedding instead of decoding it to a token [@hao2025coconut]. These methods alter training so that a model learns how to use a continuous scratchpad.

- **Construct soft tokens.** Soft Thinking passes a probability-weighted mixture of token embeddings forward, preserving uncertainty across several concepts without additional training [@zhang2025soft]. The continuous state remains tied to the vocabulary simplex.

- **Optimize activations.** PPLM established an earlier precedent for differentiating an attribute objective into a frozen language model's activations [@dathathri2020pplm]. Soft and prefix tuning optimize continuous prompts, but ordinarily learn a reusable prompt from a dataset [@li2021prefix]. Test-time latent reasoning instead learns an ephemeral state for one input from feedback available at inference time.

These distinctions are easy to blur. “Latent reasoning” can mean a learned recurrent continuous thought, a soft mixture of token embeddings, ordinary implicit computation in hidden states, or an explicitly optimized activation. Here we focus on the last meaning.

## A common mathematical frame

Let $c$ be a problem, $x=(x_1,\ldots,x_T)$ a generated reasoning trajectory, and $\pi_\theta$ a frozen autoregressive model. Ordinary generation samples

$$
\pi_\theta(x\mid c)=\prod_{t=1}^{T}\pi_\theta(x_t\mid x_{< t},c).
$$

Test-time reasoning introduces a score $R(x,c)$ and spends an inference budget searching for a high-scoring trajectory [@snell2025scaling]. Best-of-$N$ searches by drawing $N$ leaves. Tree search explores prefixes. Test-time latent reasoning introduces continuous, instance-specific variables $z$ and instead solves

$$
z^* = \arg\max_z\; \mathbb{E}_{x\sim\pi_\theta(\cdot\mid z,c)}[R(x,c)].
$$

The parameters $\theta$ never change. This equation is the article's title written in symbols: reasoning about a single problem becomes optimization over a latent space, with the reward supplying the surface to climb. The interesting questions are all hidden inside $z$:

1. Where in the network does $z$ live?
2. How does it affect the generated trajectory?
3. How does a sequence-level reward assign credit back to it?
4. What prevents optimization from leaving the model's familiar activation manifold?

## LatentSeek: search from the output side

For each position $t$, a Transformer produces a final hidden state $z_t$ immediately before the language-model head. Ordinarily, the head turns $z_t$ into logits and samples $x_t$. LatentSeek [@li2025latentseek] cuts the computational graph at this interface and treats a prefix of final hidden states as independent, optimizable variables.

It begins with an ordinary CoT rollout. If that rollout contains $T$ tokens, the method keeps roughly the first $N=\rho T$ hidden states, where $\rho$ is a fractional optimization ratio. These states provide a semantically informed initialization. They are then decoded independently through the frozen LM head:

$$
\pi(x\mid z,c)
=
\underbrace{\prod_{t=1}^{N}\pi_{\text{head}}(x_t\mid z_t)}_{\text{decode optimized latents}}
\underbrace{\prod_{t=N+1}^{T}\pi_\theta(x_t\mid x_{< t},c)}_{\text{continue autoregressively}}.
$$

After the latent prefix is converted into tokens, the model finishes the response in the usual way. A self-reward prompt asks the same model to score the solution. LatentSeek then applies a REINFORCE-style update [@williams1992reinforce]:

$$
z_t \leftarrow z_t + \eta\,
\mathbb{E}_{x\sim\pi(\cdot\mid z,c)}
\left[R(x,c)\nabla_{z_t}\log \pi_{\text{head}}(x_t\mid z_t)\right].
$$

This formulation has an appealing property: the search variable is continuous, but the reward can be arbitrary. It need not be differentiable. Reward-weighted log-probability gradients tell each $z_t$ how to make its sampled token more or less likely.

The independence assumption is consequential. It prevents the first latent from monopolizing the update through the autoregressive chain and expands the effective search surface. But it also means that the gradient reaching $z_t$ describes its influence on the token decoded _at the same position_. The influence of that token on later reasoning must pass through a discrete sample. A terminal reward can say that the trajectory was good; it cannot cleanly say which early latent altered which later inference.

That is the credit-assignment gap.

<figure class="l-page">
  <img src="/images/LatentSeek.jpg" alt="LatentSeek">
  <figcaption><strong>Figure 2.</strong> LatentSeek [@li2025latentseek]. An initial CoT supplies final-layer hidden states. A prefix of those states is optimized with reward-weighted gradients through the LM head, decoded into tokens, and followed by an ordinary autoregressive continuation. The tokenized prefix is both a useful interface and a discrete bottleneck.</figcaption>
</figure>

Across GSM8K, MATH-500, and AIME 2024, LatentSeek reports consistent gains over CoT and several sampling or reflection baselines, with improvements that continue as the allowed number of latent updates grows. With a perfect answer verifier, the average gain over CoT becomes much larger, suggesting that the search space contains better solutions than the practical self-reward mechanism can reliably identify.

The qualitative cases are at least as interesting as the aggregate scores. Some optimized prefixes become grammatically broken or semantically opaque while still leading to the correct answer. This may indicate that a model has useful computational paths that do not resemble explanations humans would write. It may also indicate reward exploitation, accidental shortcutting, or off-manifold activation. Correctness alone cannot distinguish these stories.

<figure class="l-page">
  <code><span style="color:#9c0000">QUESTION:</span> Two trains leave San Rafael at the same time [...] What’s the distance covered by each train in the two days? <br>
<span style="color:#9c0000">ANSWER:</span> 230 <br>
<span style="color:#9c0000">CoT:</span> To solve this problem, we need to break it down into steps. [...] So, the distance covered by each train in the two days is 197.5 miles. \boxed{197.5} <br>
<span style="color:#9c0000">LatentSeek:</span> Let find this„ let’ll more understand it <span style="color:blue">down step two andLet</span> 1: BothThe the trains are same route west both <span style="color:blue">first time on80ward)</span> on same and and can consider they the travel travel same distance of So’s denote the common xd’ they both 80 miles on x have write up an following: [...] Since both trains travel the same distance in each direction, the distance covered by each train is 230 miles. The final answer is: \boxed{230}
  </code>
  <figcaption><strong>Figure 3.</strong> An example of LatentSeek output [@li2025latentseek]. Despite generating linguistically anomalous expressions, the model still arrives at the correct answer.</figcaption>
</figure>

This is the productive ambiguity in the phrase “seek in the dark”: latent search is less constrained by language, but language was also our main window into what the model was doing.

## GradCuit: make the whole continuation differentiable

GradCuit [@yu2026gradcuit] changes where the latent variables enter the graph. Choose an intermediate layer $\ell$ in an $M$-layer decoder. Run the prompt and previously generated tokens through layers $1{:}\ell$, insert $N$ optimizable latent states, and pass the concatenated sequence through layers $\ell{+}1{:}M$:

$$
\pi(x_t\mid x_{< t},z^{(\ell)},c)
=
\operatorname{LMHead}\!\left(
\operatorname{Transformer}_{\ell+1:M}
\left[h_c^{(\ell)},z^{(\ell)},h_{x_{< t}}^{(\ell)}\right]
\right).
$$

Because the inserted states precede the continuation, causal self-attention lets every later token attend to every latent. The full trajectory now factorizes as

$$
\pi(x\mid z^{(\ell)},c)
=\prod_{t=1}^{T}\pi(x_t\mid x_{< t},z^{(\ell)},c),
$$

and the gradient for latent $z_i^{(\ell)}$ aggregates direct contributions from the entire continuation:

$$
\nabla_{z_i^{(\ell)}}J
=
\sum_{t=1}^{T}
\mathbb{E}\left[
R(x,c)\,
\nabla_{z_i^{(\ell)}}
\log\pi(x_t\mid x_{< t},z^{(\ell)},c)
\right].
$$

The reward is still sequence-level. The estimator is still policy-gradient-like. But the _path_ is different: a later token can assign gradient directly to an earlier latent through the remaining attention blocks. There is no latent-to-token-to-latent handoff before the continuation can use the optimized state.

This is why “circuit” is more than branding. The same self-attention graph serves as a forward computational route and a backward credit-assignment route.

<figure class="l-page">
<img src="/images/gradcuit.png" alt="GradCuit">
  <figcaption><strong>Figure 4.</strong> GradCuit [@yu2026gradcuit]. Optimizable states are inserted at layer $\ell$ between prompt states and continuation states. Forward attention broadcasts their influence to later tokens; reverse-mode differentiation carries token-level credit back along the same paths. Model weights remain frozen.</figcaption>
</figure>

Across five instruction-tuned backbones, three benchmarks, and two answer formats, GradCuit reports 64.5% average accuracy: 6.6 percentage points above standard CoT and 2.4 points above the strongest enhanced-reasoning baseline in the study. Across seven learning rates, it reduces the standard deviation of accuracy from 1.53 for LatentSeek to 0.82.

An especially diagnostic ablation replaces the reward gradient with a Gaussian random direction. This random-walk variant remains competitive with LatentSeek, while reward-guided GradCuit adds another 2.4 points on average over random optimization. The result suggests two effects:

- inserting and perturbing a state at a useful intermediate layer already opens alternative trajectories;
- reward-aligned direction supplies additional, decisive information.

The distinction matters. It prevents us from attributing every gain to precise credit assignment when some comes from changing the inference interface itself.

GradCuit also measures, for each continuation token, the norm of its gradient with respect to all optimized latents. “Because,” “therefore,” “then,” and similar reasoning connectors receive the strongest gradients across GPQA-Diamond, GSM8K, and MATH-500. One interpretation is that optimized latents primarily change how the model moves between reasoning steps rather than rewriting all content uniformly. This is a first-order sensitivity result, not a complete mechanistic explanation, but it gives us a testable picture of where latent control enters the trace.

## Jacobian Lens: reading the downstream-facing subspace

The Jacobian Lens [@gurnee2026jlens] begins from a different question: at an intermediate layer, which directions encode content the model could later put into words?

The ordinary logit lens applies the final unembedding matrix directly to an intermediate residual stream. This implicitly assumes that early and late layers use the same coordinates. The J-lens corrects for how representations are transformed by later layers. For layer $\ell$, it computes a corpus-averaged Jacobian $J_\ell$ mapping perturbations of the layer-$\ell$ residual stream to the final residual stream. A simplified readout is

$$
\operatorname{JLens}_\ell(h_\ell)
=
\operatorname{softmax}
\left(W_U\,\operatorname{norm}(J_\ell h_\ell)\right),
$$

where $W_U$ is the model's unembedding matrix. The rows of $W_UJ_\ell$ define token-indexed directions: patterns of intermediate activity that are disposed, across contexts, to make a token more likely to be verbalized later.

Anthropic calls the subspace captured by these directions **J-space**. Their experiments find that it carries silent intermediate results, planned rhyme words, situational assessments, and concepts that can be reported or causally swapped. Workspace-like content appears primarily in a middle band of layers; early layers are closer to a sensory regime, while the final layers become dominated by imminent output. J-space has limited capacity—on the order of tens of concepts at a time—and explains only a modest fraction of total activation variance [@gurnee2026jlens].

<figure class="l-page">
  <img src="/images/jlens.png" alt="JLens">
  <figcaption><strong>Figure 5.</strong> The J-lens averages a layer-to-output Jacobian across contexts to build a reusable, token-indexed readout of verbalizable directions. (Image source: @gurnee2026jlens)</figcaption>
</figure>

The resonance with GradCuit is real:

1. **Both privilege downstream influence over decodability at the current layer.** A raw hidden state matters because of what later computation will do with it.
2. **Both use Jacobian structure.** J-lens averages a linearized layer-to-output map; GradCuit differentiates the current continuation through later blocks to obtain an instance-specific update.
3. **Both point to the middle of the network.** J-lens finds a workspace-like intermediate band. GradCuit's layer ablations find early-to-middle placements—roughly 25% to 50% depth—most effective, with task dependence.
4. **Both expose transition structure.** GradCuit finds high sensitivity on reasoning connectors. J-lens observes intermediate concepts becoming available and then being routed into downstream computation.

But the objects should not be conflated.

| Property            | Jacobian Lens                                  | GradCuit                                             |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Primary goal        | Interpret and causally probe hidden content    | Improve one answer at test time                      |
| Jacobian            | Averaged over positions and a corpus           | Local to the current instance and sampled trajectory |
| Output              | Ranked vocabulary tokens / concept directions  | Reward gradient over latent states                   |
| State change        | Optional targeted read/write intervention      | Iterative unconstrained latent optimization          |
| Semantic constraint | Token-indexed verbalizable directions          | No requirement that updates be verbalizable          |
| Reuse               | Lens matrix is precomputed per model and layer | Latents are discarded after each instance            |

## The emerging family

Test-time latent reasoning is already branching by where the state is inserted and how reward is obtained.

| Method     | Optimized object                                  | Feedback                | Key distinction                                                              |
| ---------- | ------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| LatentSeek | Final-layer states for an initial output prefix   | Self-reward or verifier | Decodes optimized states before continuation                                 |
| LTPO       | Input-side latent thought vectors                 | Model confidence        | Uses Gaussian perturbations and zeroth-order policy gradients [@ye2026ltpo]  |
| GradCuit   | Prefix states at a selected intermediate layer    | Self-reward             | Assigns gradients from the full continuation through attention               |
| MILR       | Joint text and image output-side states           | Image-quality critic    | Extends latent search to unified multimodal generation [@mi2026milr]         |
| DMLR       | Latent think tokens plus selected visual features | Confidence              | Interleaves latent optimization with dynamic visual retrieval [@liu2026dmlr] |

This table reveals that “latent” is not one location. Input embeddings are easy to inject but far from the output objective. Final-layer states are close to the vocabulary but have little downstream network left to transform them. Intermediate states retain both contextual meaning and computational runway. GradCuit's empirical preference for early-to-middle layers is therefore plausible: the representation is formed enough to optimize, yet enough layers remain to absorb and refine the intervention.

The same trade-off appears in interpretability. Early states may not yet expose the relevant concept; late states may only encode the answer already chosen. The middle is where a thought can be both abstract and consequential.

## What would establish a scaling law?

The phrase _test-time scaling_ should mean more than “performance improved when we ran more steps.” A convincing latent scaling law would describe performance as a function of at least four coupled budgets:

$$
\text{quality}
=f(\text{latent dimension},\ \text{samples per update},\ \text{update steps},\ \text{reward cost}).
$$

It should compare against discrete search at matched total compute, separate search quality from verifier quality, and report when extra optimization begins to overfit the reward. It should also adapt budget per instance. A promising controller might stop when reward improvement saturates, gradient directions become unstable, or independent verifiers disagree.

Several research questions follow.

- **What should be optimized?** A free prefix, a recurrent state, a low-rank subspace, or a sparse set of concept directions?
- **Where should it live?** Can layer selection be predicted from Jacobian conditioning, workspace onset, or task uncertainty rather than tuned on a benchmark?
- **How should credit be estimated?** Full backpropagation is informative but costly; perturbation methods are cheaper in memory but noisy. Hybrid low-rank estimators may offer a better frontier.
- **How can search remain on-manifold?** Trust regions around ordinary activations, learned latent priors, or J-space-aware regularization could reduce pathological states.
- **Who checks the checker?** Diverse verifiers, process-level feedback, executable tests, and uncertainty-aware aggregation may be more important than another optimization step.
- **Can latent reasoning remain auditable?** J-lens-style readouts, causal interventions, and projection analyses could track whether gains come from recognizable intermediate computation or opaque shortcuts.

## A different kind of adaptation

The deepest idea in this line of work is not that hidden states are mysterious thoughts. It is that inference need not be a one-way execution of fixed weights.

LatentSeek turns final hidden states into per-instance policy variables. GradCuit embeds those variables inside the Transformer's computation so that the full continuation can assign them credit. The Jacobian Lens shows, from the interpretability side, that some intermediate directions are organized around information the model can later verbalize, control, and use flexibly. Together, these results suggest that intermediate representations are not merely transient by-products. They can be interfaces for search, control, and observation.

But current research still needs reliable objectives, compute-matched scaling curves, safeguards against off-manifold reward exploitation, and evidence beyond compact verifiable tasks. The most interesting future systems may combine the strengths of both worlds: the expressive freedom of continuous optimization and the auditability of a readable workspace.

Test-time latent reasoning asks a simple question with a surprisingly large design space: when a frozen model gets one problem wrong, can we improve _how it thinks about that problem_ without rewriting what it knows? The wager behind reasoning-as-optimization is that the answer turns less on the optimizer than on the space it moves through. A great latent space is one where complex problems yield to simple gradient ascent—and the work ahead is learning to find, and to shape, such spaces.

## Citation

```bibtex
@misc{li2026ttlr,
  author = {Li, Hengli and Zheng, Zilong and Zhang, Chi and Zhu, Song-Chun and Wu, Ying Nian},
  title = {Reasoning as Latent-Space Optimization},
  year = {2026},
  url = {https://latentreasoning.github.io/test-time-latent-reasoning}
}
```
