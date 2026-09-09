# Titles + Opens — Dataloading Video Pair

## Video 1

**Title (pick one):**
- PyTorch DataLoader Explained: Map vs Iterable Datasets (with a real example)
- PyTorch Dataloading: Map vs Iterable Datasets, DataLoader, and num_workers Explained
- How PyTorch Dataloading Actually Works: Dataset, DataLoader, Map vs Iterable

**Open (read nearly verbatim, ~30s):**

> Today we're building the data loading for a small time-series transformer — replacing a hand-rolled batching loop with PyTorch's real machinery.
>
> By the end of this video you'll understand:
> 1. The map — what actually happens between your data and your model: Dataset, sampler, DataLoader, collate, batch.
> 2. The two contracts a Dataset can fulfill — map-style and iterable-style — and how to choose.
> 3. What `num_workers` really does — and why iterable datasets silently duplicate your data if you're not careful.
>
> One idea to hold the whole time: the DataLoader doesn't load your data. Your Dataset does. Everything else is machinery around it.

*(Then: the diagram, fully drawn, one pass over it. Then the notebook.)*

## Video 2

**Title (pick one):**
- torchdata.nodes Explained: PyTorch Is Unbundling the DataLoader
- The New PyTorch Data Loading: torchdata.nodes from Scratch
- torchdata.nodes Tutorial: Rebuild the PyTorch DataLoader as Composable Nodes

**Open (~30s):**

> Last video we built PyTorch dataloading the classic way. Today we're taking the DataLoader apart — rebuilding the exact same pipeline with torchdata.nodes, PyTorch's new composable data loading.
>
> By the end you'll understand:
> 1. Why the DataLoader is being unbundled — the four pains of the opaque box.
> 2. The rebuild — our loader from last time, one node per line of code.
> 3. The feature the DataLoader never had — checkpointing your data pipeline mid-epoch and resuming exactly, even on an infinite stream.
>
> Same diagram as last time — but now we open the box in the middle.

*(Then: state B of the diagram. Then the notebook.)*

**Note on naming:** "NanoTST" stays out of the titles (made-up name, zero search volume). In the open it's just "a small time-series transformer I built from scratch — link below," which points people to that series without costing the title anything.

---

## Video 1 — script skeleton (Chris's structure, tightened)

1. **Today:** dataloading in PyTorch — how it fits the training loop (code), map vs iterable datasets (diagram), then run it on a toy time-series transformer (web UI).
2. **The training loop.** A batch = a bunch of samples packed into one flat tensor so the GPU can parallelize. That's one train step. Today is about everything left of the model: where that batch comes from.
3. **Scope beat:** assume your data is already preprocessed and sitting on disk/s3, ready. How it got there — and what physically happens to the bytes — is a separate video.
4. **The two classes.** The Dataset does exactly one thing: hand back a *single sample*. That's the whole contract. The DataLoader is everything around it — what **order** to ask in (sampler), how **many** per batch, how to **stack** them into one tensor (collate), how many **workers** fetch in parallel. Hand the Dataset to the DataLoader, out comes the batch tensor. *The DataLoader never loads your data. Your Dataset does.*
5. **Two flavors of Dataset:**
   - **Map** — for when you can cheaply grab sample number *i* (in memory, memory-mapped, indexed files). NOT just "fits in memory" — the rule is cheap random access.
   - **Iterable** — for when you can't: streams, huge datasets, pre-packed shards you stream in.
   - **Decision rule (say it): map until you can't.**
6. → notebook (map build → train → iterable build → worker duplication demo) → web UI training as the visual close.




