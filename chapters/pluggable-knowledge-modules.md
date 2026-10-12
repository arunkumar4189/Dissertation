# Pluggable Knowledge Modules

A frozen language model stores what it knew at training time. After that, new facts, private corpora, and domain skills have to be attached from outside. The usual attachment is retrieval-augmented generation: a retriever finds passages and the model reads them in the prompt. That works, and it is easy to update, but the knowledge never enters the same place the model uses for its own facts. Longer contexts also cost more and, past a point, the model uses them less well.

This chapter treats the alternative as a design problem. A **pluggable knowledge module** is a small, reversible attachment that carries knowledge the base model did not have, can be added or removed without retraining that model, and can be selected or combined per query. Five lines of work from 2023–2025 come closest to that definition. They differ in where the module lives, how it is built, and how several modules are combined.

## What makes a module pluggable

Four properties separate a pluggable module from ordinary fine-tuning and from a longer prompt.

1. **The base weights stay put.** The attachment is a LoRA, a set of key–value vectors, or a memory table. Removing it restores the original model.
2. **The unit of knowledge is explicit.** A module corresponds to a document, a triple, a domain, or a skill, so it can be indexed, versioned, and deleted.
3. **Attachment is per query.** The system chooses which modules to load for this question, then unloads them.
4. **Composition is defined.** Two modules have a rule for how they occupy the same model at once: a sum, a gate, or joint attention.

In-context RAG satisfies (2) and (3) and fails (1) in spirit: the passage is present only as tokens, and attention over those tokens grows with the square of the combined length. Full fine-tuning satisfies none of the four. The methods below trade off the four properties differently.

## Where the knowledge sits

| Family | What is stored | When it is built | How a query uses it | Representative work |
| --- | --- | --- | --- | --- |
| Document LoRA | One low-rank update per document | Offline, one training run per document | Retrieve updates and add them into the feed-forward weights | Parametric RAG (Su et al., SIGIR 2025) |
| Generated LoRA | A hypernetwork that emits a LoRA | One translator, trained once | Encode the retrieved document and generate its update | DyPRAG (Tan et al., 2025) |
| Knowledge tokens | One key–value pair per triple | Encode with a frozen sentence encoder and a small adapter | Rectangular attention from prompt tokens onto the knowledge tokens | KBLaM (Wang et al., ICLR 2025) |
| Routed domain LoRAs | One LoRA per task or domain | Ordinary LoRA fine-tuning, done ahead of time | Choose or mix coefficients, often from a few examples | LoRAHub (Huang et al., 2023); MoLE (Wu et al., 2024) |
| Memory layer | A large trainable key–value table inside selected layers | Pretraining or continued training | Top-$k$ lookup, sparse in parameters and cheap in FLOPs | Memory Layers at Scale (Berges et al., 2025) |

The first four families leave the pretrained model frozen and attach knowledge later. Memory layers are different: the table is trained with the model, and the “plug” is a slot in the architecture rather than a cartridge swapped at inference. They belong in the same chapter because they are the scaled version of the same idea, a dedicated store the network reads instead of a longer prompt.

## Parametric RAG: one LoRA per document

Su et al. (2025) start from a mismatch. Standard RAG appends retrieved text to the input, which they call in-context knowledge injection. The model’s own knowledge sits in its parameters, especially in the feed-forward networks. Putting a passage in the prompt never closes that gap, and the cost grows with every extra passage.

Parametric RAG moves the injection into those feed-forward weights. Offline, each document $d_i$ is turned into a parametric representation $p_i$, a few megabytes of LoRA matrices on the feed-forward layers (Hu et al., 2022). The paper prefers LoRA over adapters and prefix tuning because LoRA merges by addition and adds little cost at inference. Online, the pipeline is Retrieve–Update–Generate:

1. **Retrieve.** A standard retriever returns the top-$k$ documents for the query.
2. **Update.** Their LoRA factors are merged and added to the frozen weights.
3. **Generate.** The updated model answers the original query, with no passages in the prompt.

For a feed-forward matrix $W$, document $j$ contributes low-rank factors whose product is the update. The merged update is a scaled sum,

$$
\Delta W_{\text{merge}} = \alpha \sum_{j=1}^{k} A_j B_j^{\top}, \qquad W' = W + \Delta W_{\text{merge}}.
$$

Su et al. report that loading these parameters is on the order of one percent of the cost of decoding a single token. The same system can still place the passages in the prompt; they find the combination stronger than either injection alone.

The price is paid offline. Every document needs its own augmentation and its own short training run, and the corpus of LoRAs has to be stored. That cost is linear in the number of documents and is paid again if the base model changes. Merge-by-sum also assumes that independently trained updates add rather than interfere. Nothing in the merge step rescales a document that conflicts with another retrieved document or with the base model.

## DyPRAG: generate the LoRA instead of storing it

Tan et al. (2025) keep the same injection site and remove the per-document training loop. They treat Parametric RAG as a search for a mapping $\mathcal{F}$ from a document to a parameter update, and they fit that mapping once. A small hypernetwork, the **parameter translator** $\mathcal{F}'_\phi$, reads a document embedding and writes the LoRA factors.

Training has two stages. The first is the Parametric RAG procedure on a set of documents, which produces document–parameter pairs. The second trains the translator to reproduce those parameters from the document’s last-token hidden state, $s_i \in \mathbb{R}^{h}$, before the vocabulary projection. The translator is a short stack of linear layers. At test time it emits LoRA factors for the feed-forward projections (`up_proj`, `down_proj`, `gate_proj` in the reference implementation), the factors for the retrieved documents are combined, and the base model generates.

The storage story changes completely. Parametric RAG keeps one adapter per document. DyPRAG keeps one translator. A new document does not need its own optimisation run; encoding it and running the translator is enough. Tan et al. argue that this also helps when the prompt and the model’s parameters disagree, because the generated update puts the retrieved fact into the same weights the model uses for its own knowledge.

The translator can only emit updates that look like the ones it was trained to imitate. A domain far from that alignment set, or a base model other than the one whose hidden states and LoRA layout the translator saw, is outside what the published method demonstrates. The per-document LoRAs are still required to build the alignment set, so the offline cost moves from the full corpus to a training corpus for $\mathcal{F}'$.

## KBLaM: knowledge as tokens the model attends to

Wang, Isazawa, Mikaelyan, and Hensman (2025) do not modify weights at all. KBLaM encodes a knowledge base of triples and inserts the encodings into attention. Each triple becomes a **knowledge token**: a key vector and a value vector of the same width as one entry in the model’s key–value cache. A pretrained sentence encoder produces the base embeddings; learned linear adapters map them into the model’s key and value spaces. The key side is an index, built from the entity and the property. The value side carries the property value.

The attention pattern is rectangular. Prompt tokens attend to every knowledge token. Knowledge tokens do not attend to each other, and they do not attend back to the prompt. The extra cost is linear in the number of triples, where full self-attention over the same material would be quadratic. Wang et al. integrate more than 10,000 triples into an 8B model whose context window is 8,000 tokens, on one A100 80GB GPU. Because each triple is encoded independently, editing the knowledge base means re-encoding the changed triples. The base model is not fine-tuned to absorb the edit.

Rectangular attention is also the retrieval step. There is no separate index: the query hidden states score the knowledge keys. Attention weights over keys are a readable trace of which triples the layer used. The independence assumption is stated plainly by the authors. Facts that only make sense together are a weak fit for tokens that cannot see each other.

KBLaM is pluggable at the level of a whole knowledge base, and at the level of a single triple inside it. It is a poorer fit for long unstructured documents, unless those documents are first decomposed into triples.

## LoRAHub and mixture-of-LoRA routing

A third granularity sits between a single fact and a whole pretraining corpus: a skill or a domain, already packaged as a LoRA.

**LoRAHub** (Huang et al., 2023) takes LoRAs trained on different tasks and composes them for a task none of them was trained on. Composition is a weighted sum of the adapters. The coefficients are fit on a handful of target examples with a gradient-free optimiser, so the base model and the LoRAs stay frozen. After a few iterations the composed adapter is merged in and used as usual. The paper’s picture of mixture-of-experts is useful and inexact: the “experts” were trained separately, the number of them is not fixed at training time, and the router runs once per task, over coefficients, with no per-token gate.

**Mixture of LoRA Experts** (Wu, Huang, and Wei, 2024) starts from the failure mode of that sum. One scalar per LoRA, shared by every layer, washes out the adapter once several of them are added. MoLE treats each layer of each LoRA as its own expert and learns a gate inside the layer. Only the gates are trained. At inference the learned gates can be used as they are, or selected LoRAs can be masked and the remaining weights renormalised, without a second training run. The same recipe is demonstrated for language and for vision–language adapters.

These routers solve a different problem from Parametric RAG. The module is a domain, not a passage, and the selection is over a small catalogue of adapters the system already trusts. They do not ingest a new document at query time. They matter here because any document-level design eventually faces the same composition question: once several modules are loaded, equal weights are a blunt instrument, and a learned gate is one answer.

## Memory layers: a table inside the network

Berges et al. (2025), from Meta FAIR, scale an older idea. A memory layer replaces a feed-forward layer with a lookup. A query vector scores a large set of trainable keys, the top $k$ keys are kept, and the output is a weighted sum of the corresponding trainable values. The table can hold orders of magnitude more parameters than the dense layer it replaces, while the FLOPs stay close to a top-$k$ retrieval plus a small aggregation. Product keys (Lample et al., 2019) make the search practical: two half-key codebooks of size $\sqrt{N}$ stand in for $N$ full keys that are never materialised. Values are sharded across GPUs. Berges et al. share one memory pool across several layers and find that about three memory layers help, after which swapping out more feed-forward layers hurts. Sparse memory and dense computation are complements.

The empirical claim is about facts. Models with the memory layer, pretrained for a trillion tokens with up to 128 billion memory parameters, beat dense models trained with more than twice the compute, and beat mixture-of-experts models matched for compute and parameter count. The gains are largest on factual question answering. The memory is still trained end to end. It is not a cartridge you encode from a PDF and slot in after deployment. Forgetting or editing one fact means changing a distributed table that was optimised jointly with the rest of the network. Product-key memory and the later PEER architecture (He, 2024), which stores rank-one experts instead of vectors, show how large that table can get. They leave the plug-in interface open.

## Comparison

Six questions decide which design is actually pluggable for a given corpus.

**Unit.** Parametric RAG and DyPRAG use the document. KBLaM uses the triple. LoRAHub and MoLE use the task or domain. Memory layers use a key that the training loss found useful, which usually does not line up with a source the user can point to.

**Construction.** Parametric RAG trains one adapter per document. DyPRAG trains one translator, then generates adapters. KBLaM encodes with a frozen encoder and a small adapter. LoRAHub assumes the adapters already exist and only tunes coefficients. Memory layers require a pretraining run.

**Edit.** Deleting a document in Parametric RAG deletes its file of weights. In DyPRAG there may be nothing to delete, because the adapter was generated for that query. In KBLaM the triple is re-encoded or dropped. In a memory layer an edit is a research problem of its own.

**Conflict.** In-context RAG can place two passages side by side and hope the model notices the contradiction. A summed LoRA has no such channel. KBLaM at least exposes attention mass over keys. None of the five methods has a dedicated conflict policy.

**Cost shape.** Parametric RAG spends compute offline and keeps inference short. DyPRAG spends less storage and shifts work to the translator at test time. KBLaM spends a linear attention over the knowledge base on every layer where the tokens are injected. Memory layers spend parameters, not FLOPs, and spend them during training.

**Coupling to one base model.** A LoRA is tied to the shapes of the matrices it updates. A KBLaM adapter is tied to the key–value width and to the layers it was trained to write into. A memory layer is part of the model. Swapping the backbone means rebuilding the modules. That coupling is the main practical limit on a shared library of knowledge modules.

## Open problems

The literature shows that external knowledge can enter a model as parameters, as attention keys, or as a sparse table. It does not yet show a module format that is at once document-level, cheap to build, composable under conflict, and portable across base models. Four gaps follow directly from the designs above.

**Interference under merge.** Summing LoRAs is the composition rule in Parametric RAG and the starting point for LoRAHub. MoLE shows that a single scalar per adapter is already too coarse for domain modules. Document modules are more numerous and less curated. A composition rule that can down-weight a module the query has already contradicted would change when parametric injection is safe to use.

**A module the user can delete.** Memory layers store facts well and do not name them. KBLaM names them and assumes they are triples. A document-level module with KBLaM’s edit semantics, or a memory table whose keys can be traced back to sources, would make retention and deletion ordinary operations.

**Translators that survive a new corpus.** DyPRAG is the existence proof that a hypernetwork can imitate document LoRAs. How far that translator travels, across domains and across model families, is the difference between a research pipeline and a format for publishing knowledge.

**One index, two injection paths.** Every LoRA method still uses an external retriever, then a second mechanism to apply what was retrieved. KBLaM folds retrieval into attention and gives up unstructured documents. A system that retrieves once and can inject the hit either as a LoRA or as a knowledge token, chosen by the shape of the source, is the architecture this chapter is pointing at.

That architecture is the working target of this dissertation: knowledge published as modules, attached for one query, and detached without a trace in the base model.
