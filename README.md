<div align="center">

# Rethinking Continual Experience Internalization for Self-Evolving LLM Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.04703)

</div>

Official code for the paper *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents*.

The pipeline turns an agent's interaction trajectories into a compact, **principle-level experience pool** and internalizes it into the model parameters, so the agent and its experience co-evolve across iterations.

## Components

- **[`experience_extraction/`](experience_extraction/)** — agent rollout + experience summarization: a local model solves tasks (web reasoning / math), and the scored trajectories are distilled into a reusable experience pool. See its [README](experience_extraction/README.md) for setup and usage.
- **[`evaluation/`](evaluation/)** — web-agent evaluation on WebWalkerQA, GAIA-Text-103, and BrowseComp-ZH, with and without inference-time experience injection. Built on the Tongyi DeepResearch inference framework. See its [README](evaluation/README.md).

More components (experience injection and context-distillation training) will be released here.
