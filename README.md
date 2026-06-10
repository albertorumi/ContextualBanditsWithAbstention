# Bandits with Abstention under Expert Advice

Code for the NeurIPS 2024 paper [*Bandits with Abstention under Expert Advice*](https://arxiv.org/abs/2402.14585)
(Pasteris, Rumi, Thiessen, Saito, Miyauchi, Vitale, Herbster).

We study prediction with expert advice under bandit feedback when one action — abstention — incurs no reward
or loss. The proposed CBA algorithm exploits this assumption to obtain reward bounds that improve on the
classical EXP4 approach. The experiments in this repository evaluate CBA on online multiclass node
classification over graphs, where the experts are *specialists* (confidence-rated predictors derived from the
graph structure) and the learner may abstain on nodes where the specialists are not confident.

## Repository structure

```
src/
  bandits.py              CBA (class OSE4) and its training loop
  vcbases.py              Construction of the specialist sets ("bases") from the graph:
                          interval bases, Louvain communities (with peeling), hierarchical
                          clustering, distance balls; full-information training loops
  winnow.py               Winnow variants operating on the bases (full-information setting)
  GABA.py                 GABA baseline (adversarial bandits over a random spanning tree)
  exp3.py                 EXP3 baseline (one instance per node)
  contextualsimilarity.py Contextual bandit baseline with a similarity/zooming schedule
  baseline_models.py      Training loops for the non-bandit baselines
  knn.py, perceptron.py,
  wta.py, mv.py           Baseline predictors (k-NN, graph perceptron, WTA on a spanning
                          tree, majority vote)
  utils.py                Graph generators (Gaussian, multi-class cliques, noise injection),
                          Wilson's random spanning tree, plotting helpers
  tree_utils.py           Binary-tree helpers used by GABA
  balls.py                Distance/ball computations on graphs
  convg.py                DGL -> NetworkX conversion (Cora)

test/
  Gaussian.ipynb          Synthetic experiment: Gaussian graphs
  MultiClique.ipynb       Synthetic experiment: multi-class cliques with noise
  LastFM.ipynb            Real-data experiment: LastFM social network
  Cora.ipynb              Real-data experiment: Cora citation graph
  results/                Saved results per dataset
```

## Running the experiments

```bash
pip install -r requirements.txt
jupyter notebook test/
```

Open one of the notebooks in `test/` and run it top to bottom. Each notebook builds (or loads) its graph,
constructs the specialist bases, trains CBA and the baselines, and plots cumulative mistakes with confidence
intervals. Results are written to `test/results/`.

## Citation

```bibtex
@inproceedings{pasteris2024bandits,
  title     = {Bandits with Abstention under Expert Advice},
  author    = {Pasteris, Stephen and Rumi, Alberto and Thiessen, Maximilian and Saito, Shota
               and Miyauchi, Atsushi and Vitale, Fabio and Herbster, Mark},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
