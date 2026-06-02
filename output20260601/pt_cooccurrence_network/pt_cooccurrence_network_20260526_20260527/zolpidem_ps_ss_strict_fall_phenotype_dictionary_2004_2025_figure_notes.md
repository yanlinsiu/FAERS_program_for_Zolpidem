# PT co-occurrence network notes

- Period: `2004_2025`
- Cohort: `zolpidem_ps_ss_strict_fall`
- Term set: `phenotype_dictionary`
- Cohort cases: `1058`
- Node inclusion: top `50` PTs with at least `5` cases
- Edge inclusion: co-occurrence count at least `3`
- Edge metrics: co-occurrence count, Jaccard index, lift, and phi coefficient
- Figure edge width/layout weight: `jaccard`
- Community detection: NetworkX Louvain communities over strong retained edges with `jaccard >= 0.15`.
- Important: this is a reported-event phenotype network, not a causal mechanism graph.
