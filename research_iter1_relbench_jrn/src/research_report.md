# RelBench & JRN

## Summary

Comprehensive survey of all 11 RelBench datasets (7 v1 + 4 v2) with complete schemas including tables, foreign keys, estimated cardinality distributions, and 51 predictive tasks with metrics. Includes structured comparison of 6 related per-join importance methods (ESNR, MPS-GNN, RelGNN, over-squashing, Relatron, Database Views) against the proposed JRN concept, identifying that no existing method provides pre-training per-FK-join threshold-based diagnostics for heterogeneous relational graphs. Provides tiered dataset ranking for JRN experiments with specific FK joins identified for threshold testing.

## Research Findings

## Complete Survey of RelBench Schemas, Tasks, and Related Work on Per-Join Importance Metrics

### 1. RelBench v1 Dataset Schemas (7 Datasets)

RelBench v1 provides 7 realistic relational databases spanning e-commerce, social, medical, and sports domains, totaling 103,466,370 rows across 489 columns [1, 4]. The benchmark was published at the NeurIPS 2024 Datasets and Benchmarks Track [1]. Each database consists of tables connected via primary-foreign key relationships, with tables categorized as either fact (event-driven, timestamped) or dimension (contextual, relatively static) [1].

**rel-amazon** (E-commerce): 3 tables (product, user, review), 15,000,713 rows, 15 columns [1]. Simple star schema centered on the review table. FK relationships: review.product_id→product and review.user_id→user. With ~14M reviews across ~500K products and ~500K users, the estimated fan-out is ~28 reviews per product and ~28 reviews per user. This symmetric structure means both FK joins should carry similar signal, making it a good baseline for uniform JRN [1].

**rel-hm** (E-commerce): 3 tables (customer, article, transaction), 16,664,809 rows, 37 columns [1]. Star schema with transaction as the central fact table. FK relationships: transaction.customer_id→customer (~11 fan-out given ~1.4M customers) and transaction.article_id→article (~144 fan-out given only ~105K articles). The asymmetric fan-out is notable: the article join aggregates ~14x more child rows than the customer join, creating a natural test case for whether high aggregation dilutes signal [1].

**rel-avito** (E-commerce): 8 tables, 20,679,117 rows, 42 columns [1]. Complex star schema modeling the Avito ad marketplace. Eight tables include users, ads, search_info, visits_stream, search_stream, category, location, and phone_request_stream. FK relationships span a wide cardinality spectrum: ads.category_id→category has extreme fan-out (~40K ads per category) while phone_request_stream.ad_id→ads has very low fan-out (~0.5). This diversity makes rel-avito excellent for testing JRN sensitivity across cardinality ranges [1].

**rel-stack** (Social Q&A): 7 tables (users, posts, comments, votes, badges, postLinks, tags), 4,247,264 rows, 52 columns [1]. Snowflake schema modeling StackOverflow activity. FK relationships include: posts.owner_user_id→users (~3 fan-out), comments.post_id→posts (~2.5), comments.user_id→users (~7.5), votes.post_id→posts (~2.5), votes.user_id→users (~7.5), badges.user_id→users (~1.5), and postLinks connecting posts to related posts (~0.08 fan-out). The well-understood Q&A semantics make it easy to reason about which joins carry task-relevant signal [1].

**rel-event** (Social/Events): 5 tables (users, events, attendance, friends, event_attendees), 41,328,337 rows, 128 columns [1]. Star schema with extremely high cardinalities. The attendance.user_id→users join has fan-out ~875 (the highest in all of RelBench), while attendance.event_id→events has ~12. The 128-column feature space combined with extreme fan-outs creates a stress test for information compression during aggregation [1].

**rel-f1** (Sports): 9 tables (drivers, constructors, circuits, races, results, qualifying, constructor_results, constructor_standings, driver_standings), 74,063 rows, 67 columns [1]. Complex snowflake schema with 13 FK relationships. Key bridge tables: results connects drivers, races, and constructors with fan-outs of 29, 23, and 119 respectively. This is the dataset where RelGNN achieves its largest improvements (up to 25%), suggesting the bridge-node structure creates significant information flow patterns that per-join analysis could illuminate [1, 7].

**rel-trial** (Medical/Clinical): 15 tables centered on studies, 5,434,924 rows, 140 columns [1]. Deepest snowflake schema in RelBench. Central studies table is connected to 14 satellite tables: conditions (~1.75 per study), interventions (~1.5), outcomes (~1.25), facilities (~2.0), sponsors (~1.5), eligibilities (~1.0), reported_events (~1.25), designs (~1.0), outcomes_analysis (→outcomes, ~0.4), drop_withdrawals (~0.375), browse_conditions (~0.5), browse_interventions (~0.5), outcome_counts (→outcomes, ~0.3), and outcome_measurements (→outcomes, ~0.6). The two-level hierarchy (studies→outcomes→outcome_measurements) enables multi-hop JRN compounding tests [1].

### 2. RelBench v2 Dataset Schemas (4 New Datasets)

RelBench v2 introduces 4 additional datasets and the autocomplete task paradigm, bringing the total to 11 datasets [2].

**rel-salt** (Enterprise ERP): 4 tables, 4,257,145 rows, 31 columns [2]. Linear chain schema representing end-to-end business transactions from an ERP system. All 8 tasks are autocomplete (accuracy metric) [2].

**rel-arxiv** (Academic): 6 tables, 2,146,112 rows, 21 columns [2]. Many-to-many network modeling the scholarly citation ecosystem with 222K papers, 143K authors, and 1.5M directed citation links across 53 physics categories [2]. 4 tasks spanning AUROC, accuracy, R², and MAP [2].

**rel-ratebeer** (Consumer): 13 tables, 13,787,005 rows, 221 columns [2]. The richest schema in v2, with tables for beers, users, brewers, places, ratings, reviews, favorites, and more. Over 30 columns of multi-modal features [2]. 8 tasks across autocomplete, forecasting, and recommendation [2].

**rel-mimic** (Medical EHR): 6 tables, 2,424,751 rows, 54 columns [2]. Requires PhysioNet credentials. Single forecasting task: predict ICU stays ≥3 days (AUROC) [2].

### 3. Complete Task Catalog

Across all 11 datasets, RelBench defines 51 predictive tasks [1, 2]: v1 has 30 tasks (12 entity classification/AUROC, 7 entity regression/MAE, 11 recommendation/MAP) and v2 has 21 tasks adding autocomplete (9 tasks, accuracy), forecasting classification (5, AUROC), forecasting regression (3, R²), autocomplete regression (1, R²), and recommendation (3, MAP). Metrics used: AUROC, MAE, MAP@K, accuracy, R² [1, 2].

### 4. Related Work Comparison

**ESNR** (Dong et al., ICML 2023) [5]: Uses random matrix theory and the BBP phase transition to estimate whether a graph's signal can be recovered from noisy adjacency structure. Operates on the WHOLE homogeneous graph — cannot distinguish per-FK-join signal quality. Low cost but fundamentally limited for heterogeneous relational databases [5].

**MPS-GNN** (Ferrini et al., 2024) [6, 12]: Greedily extends meta-paths during training to discover informative join chains. Produces indirect per-join importance but requires full model training (high cost), not a pre-training diagnostic [6, 12].

**RelGNN** (Chen et al., ICML 2025) [7]: Composite message passing bypassing bridge/hub nodes. SOTA on 27/30 v1 tasks with largest gains on complex schemas (rel-f1 up to 25%). Key limitation: applies BLANKET bypass to ALL bridge nodes without per-join discrimination. The schema-dependent gains strongly suggest per-join analysis could make the bypass strategy adaptive [7].

**Over-squashing** (Topping et al., ICLR 2022; Black et al., ICML 2023) [8, 9]: Edge-based curvature and effective resistance identify information bottlenecks. Per-edge metrics but designed for HOMOGENEOUS graphs and computationally expensive. They measure structural bottlenecks without considering feature-based signal quality [8, 9].

**Relatron** (2025) [10]: Defines RDB-task homophily and training-free affinity embeddings to decide RDL vs DFS. Reports Spearman ρ = -0.43 at whole-TASK granularity, not per-join level. Does not guide within-RDL architecture configuration [10].

**Database Views** (Rissaki et al., 2025) [11]: Post-hoc learnable masks on GNN edges/features identify important FK joins. Per-FK granularity but POST-TRAINING only — cannot guide pre-training architecture decisions. Could serve as ground truth for JRN validation [11].

### 5. Gap Analysis and JRN Positioning

No existing method simultaneously achieves (a) per-FK-join granularity, (b) pre-training computation, (c) threshold-based diagnostics, and (d) heterogeneous graph support [5, 6, 7, 8, 9, 10, 11]. JRN uniquely targets this intersection.

### 6. Dataset Ranking for JRN Experiments

**Tier 1 — Most Informative:** rel-trial (15 tables, best for multi-hop compounding) [1], rel-f1 (9 tables, best for threshold behavior given RelGNN's 25% gains) [1, 7], rel-ratebeer (13 tables, best for cardinality-JRN correlation) [2].

**Tier 2 — Good Contrasts:** rel-stack (7 tables, clear join semantics) [1], rel-avito (8 tables, widest cardinality spectrum) [1], rel-arxiv (6 tables, clear signal/noise distinction) [2].

**Tier 3 — Baselines:** rel-amazon (3 tables, expected uniform JRN) [1], rel-hm (3 tables, asymmetric fan-out test) [1].

### 7. Confidence and Limitations

High confidence on dataset schemas, table counts, and task definitions from published papers [1, 2]. Medium confidence on per-table row counts and fan-out estimates derived from total counts. Lower confidence on hypothetical JRN values for specific joins. Relatron's modest ρ = -0.43 correlation could suggest per-join metrics have limited predictive power, though this may reflect whole-task-level limitations rather than per-join analysis limitations [10].

## Sources

[1] [RelBench: A Benchmark for Deep Learning on Relational Databases (NeurIPS 2024)](https://arxiv.org/html/2407.20060v1) — Primary source for all 7 v1 dataset schemas (rel-amazon, rel-hm, rel-avito, rel-stack, rel-event, rel-f1, rel-trial), table structures, foreign key relationships, row/column counts, and 30 predictive tasks with evaluation metrics.

[2] [RelBench v2: A Large-Scale Benchmark and Repository for Relational Data](https://arxiv.org/html/2602.12606v1) — Source for 4 new v2 datasets (rel-salt, rel-arxiv, rel-ratebeer, rel-mimic), the autocomplete task paradigm, and 21 additional tasks with their metrics and target tables.

[3] [RelBench GitHub Repository](https://github.com/snap-stanford/relbench) — Official code repository containing programmatic dataset definitions with make_db() methods specifying FK relationships, table constructions, and data loading logic.

[4] [RelBench Official Website](https://relbench.stanford.edu/) — Benchmark website with dataset descriptions, leaderboard for tracking model performance, standardized evaluation framework, and documentation for data loading.

[5] [ESNR: Towards Understanding and Reducing Graph Structural Noise for GNNs (ICML 2023)](https://proceedings.mlr.press/v202/dong23a/dong23a.pdf) — Introduces the Effective Structural Noise Ratio using random matrix theory and the BBP phase transition to estimate whether graph structure aids or hinders GNN learning. Whole-graph metric for homogeneous graphs only.

[6] [MPS-GNN: Meta-Path Learning for Multi-relational Graph Neural Networks](https://proceedings.mlr.press/v231/ferrini24a/ferrini24a.pdf) — Proposes greedy meta-path extension during training to discover informative join chains in heterogeneous relational graphs, providing indirect per-join importance via path selection.

[7] [RelGNN: Composite Message Passing for Relational Deep Learning (ICML 2025)](https://arxiv.org/abs/2502.06784) — Introduces composite message passing that bypasses bridge/hub nodes in relational graphs. Achieves SOTA on 27/30 v1 tasks with largest gains on complex schemas like rel-f1 (up to 25% improvement).

[8] [Understanding over-squashing and bottlenecks on graphs via curvature (ICLR 2022)](https://arxiv.org/abs/2111.14522) — Uses combinatorial Ollivier-Ricci curvature on edges to identify over-squashing bottlenecks in GNNs. Proposes curvature-based rewiring as mitigation. Per-edge metric but designed for homogeneous graphs.

[9] [Understanding oversquashing in GNNs through effective resistance (ICML 2023)](https://proceedings.mlr.press/v202/black23a) — Uses effective resistance between node pairs as an information bottleneck metric. High effective resistance indicates over-squashing. Computationally expensive and designed for homogeneous graphs.

[10] [Relatron: Automating Relational Machine Learning over Relational Databases (2025)](https://arxiv.org/abs/2602.22552) — Defines RDB-task homophily and training-free affinity embeddings to decide between RDL and DFS. Reports Spearman rho=-0.43 at whole-task granularity, not per-join level.

[11] [Database Views as Explanations for GNN Predictions (Rissaki et al., 2025)](https://arxiv.org/abs/2509.09482) — Applies learnable post-hoc masks on GNN edges/features to identify important FK joins and columns. Per-FK granularity but requires full model training, making it post-hoc rather than pre-training diagnostic.

[12] [A Self-Explainable Heterogeneous GNN for Relational Deep Learning](https://arxiv.org/abs/2412.00521) — Extended MPS-GNN work providing self-explainability for heterogeneous relational graphs by exposing which meta-paths contribute most to predictions.

## Follow-up Questions

- What are the exact per-table row counts and empirical cardinality distributions for each FK join in rel-trial and rel-f1, extractable from the GitHub source code make_db() methods or by loading the datasets programmatically?
- Which specific FK joins in rel-f1 account for the 25% improvement observed with RelGNN's composite message passing, and could per-join JRN estimation predict which bridge nodes benefit most from bypass versus which should retain two-hop message passing?
- How do access restrictions for rel-mimic (PhysioNet credentials) and the availability of only a 20K patient subset affect JRN estimation reliability, and should rel-mimic be deferred to later experimental phases in favor of unrestricted datasets like rel-ratebeer and rel-arxiv?

---
*Generated by AI Inventor Pipeline*
