# PT co-occurrence network notes

- Period: `2004_2025`
- Cohort: `zolpidem_ps_ss_strict_fall`
- Cohort cases: `1043`
- Node inclusion: top `50` PTs with at least `5` cases
- Edge inclusion: co-occurrence count at least `3`
- Edge metrics: co-occurrence count, Jaccard index, lift, and phi coefficient
- Figure edge width/layout weight: `jaccard`
- Community detection: deterministic weighted label propagation over the retained PT network.
- Important: this is a reported-event phenotype network, not a causal mechanism graph.
