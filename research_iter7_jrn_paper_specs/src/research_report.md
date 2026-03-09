# JRN Paper Specs

## Summary

Comprehensive paper writing specifications for the 'When Do Joins Help?' JRN paper with 8 deliverables: paper metadata with 3 title options and 250-word abstract, 7-section specification mapping every claim to experimental evidence and figure/table references, 6-figure plan, 9-table plan, negative results framing for the inverted-U failure and architecture guidance weakness, 5 pre-drafted reviewer rebuttals, 6 related work comparison paragraphs, and 5 contribution bullets. All grounded in verified citations for 9+ related works and precise RelBench dataset statistics for 4 databases.

## Research Findings

This research produces complete paper writing specifications for the 'When Do Joins Help?' JRN paper, encompassing 8 deliverables synthesized from verified citation details, RelBench dataset schemas, and best practices for framing negative results in machine learning.

## DELIVERABLE 1: Paper Metadata

The recommended title is 'When Do Joins Help? Diagnosing Information Flow Across Foreign Keys in Relational Deep Learning' because it honestly frames the paper as answering a diagnostic question rather than asserting a strong positive result, following the advice of Karl et al. [11] on embracing negative results and Lipton's heuristic [13] to lead with problems rather than solutions. Two alternatives are provided: one emphasizing the JRN metric itself and one emphasizing practical utility. The 250-word abstract follows the structure: problem statement (RDL treats all joins uniformly), proposal (JRN metric inspired by epidemiological R0), positive findings (Spearman rho > 0.6, multiplicative compounding R-squared > 0.5, pruning helps), negative findings (inverted-U threshold prediction fails, architecture guidance weak), and significance (honest characterization advances understanding).

## DELIVERABLE 2: Section-by-Section Specification

Seven sections are specified with claims-evidence-figure/table mapping:

**Section 1 (Introduction, ~1.5pp):** Motivating example from the rel-f1 database [14] showing high-JRN joins (e.g., results-to-drivers hub node with 3 foreign keys) versus low-JRN joins (e.g., circuits-to-races). The rel-f1 schema contains 9 tables with 97,606 rows and 77 columns, featuring hub nodes (results: 3 FKs) and bridge nodes (standings: 2 FKs) [14]. Ends with 5 contribution bullets.

**Section 2 (Related Work, ~1.5pp):** Six comparison paragraphs positioning JRN against each related work family, supported by a comparison matrix (Table 1). ESNR [1] proposes an edge signal-to-noise ratio using random matrix theory for homogeneous graphs (ICML 2023, PMLR 202:8202-8226), but measures overall graph noise rather than per-join transmission. MPS-GNN [2, 3] by Ferrini et al. learns informative meta-paths during training (LoG 2024, PMLR 231:2:1-2:17; extended in arXiv:2412.00521), but this makes it as expensive as full model training unlike JRN's pre-training probes. RelGNN [4] by Chen, Kanatsoulis, and Leskovec (ICML 2025, PMLR 267:8296-8312) achieves SOTA on 30 RelBench tasks with up to 25% improvement through composite message passing that bypasses bridge and hub nodes via atomic routes, but applies a blanket strategy rather than adaptive per-join allocation. Database Views [5] by Rissaki et al. (arXiv:2509.09482) provides post-hoc explainability via database view definitions grounded in the determinacy framework, but requires a trained model. Relatron [6] by Chen et al. (ICLR 2026, arXiv:2602.22552) meta-selects between RDL and DFS using RDB-task homophily, achieving 18.5% improvement at 10x lower cost, but operates at whole-task granularity. Over-squashing work by Topping et al. [7] (ICLR 2022, arXiv:2111.14522, Outstanding Paper Honorable Mention) introducing Balanced Forman curvature, and Di Giovanni et al. [8] (ICML 2023, PMLR 202:7865-7885, arXiv:2302.02941) showing topology dominates via commute time, both analyze the full data graph rather than schema-level join types.

**Section 3 (Method, ~2pp):** Formal JRN definition as performance ratio, probe specification (32-dim embeddings, 10 epochs, mean aggregation), multiplicative compounding model for multi-hop chains, and allocation algorithm pseudocode. The GNN-VPA work [9] by Schneckenreiter et al. (ICLR 2024 Tiny Papers, arXiv:2403.04747) on variance-preserving aggregation is referenced when discussing how different aggregation strategies affect signal transmission variance.

**Section 4 (Experimental Setup, ~1pp):** Four RelBench [10] databases providing diverse coverage. Robinson et al. [10] (NeurIPS 2024, arXiv:2407.20060) established RelBench with 7 databases in v1 (now 11 in v2) and 66 total tasks with standardized temporal splits and evaluation metrics. The four selected databases are: rel-f1 [14] (motorsport, 9 tables, ~74K rows, 67 columns, 6 tasks including driver-dnf/AUROC, driver-top3/AUROC, driver-position/MAE), rel-stack [15] (social Q&A, 7 tables, ~4.2M rows, 52 columns, 6 tasks including user-engagement/AUROC, post-votes/MAE, user-post-comment/MAP), rel-amazon [16] (e-commerce, 3 tables, ~15M rows, 15 columns, 8 tasks including user-churn/AUROC, user-ltv/MAE, user-item-purchase/MAP), and rel-hm [17] (fashion retail, 3 tables, ~16.7M rows, 37 columns, 4 tasks including user-churn/AUROC, item-sales/MAE, user-item-purchase/MAP). Together these span scales from 74K to 16.7M rows and join structures from simple 3-table stars (rel-amazon [16], rel-hm [17]) to complex 9-table schemas with hub and bridge nodes (rel-f1 [14]).

**Section 5 (Results, ~2.5pp):** Five sub-sections: (5.1) JRN validation with probe-vs-full scatter and meta-analytic forest plot showing per-dataset Spearman correlations with 95% CIs, pooled random-effects estimate, I-squared statistic, and prediction interval; (5.2) threshold prediction negative result showing aggregation variance increases monotonically with JRN rather than peaking at JRN approximately 1; (5.3) multiplicative compounding validation for multi-hop chains; (5.4) main architecture comparison showing JRN-pruned matches or exceeds uniform on most tasks while JRN-guided aggregation selection adds less than 0.5 percentage points; (5.5) cross-task transfer showing JRN rankings correlate at rho > 0.7 within databases.

**Section 6 (Discussion, ~1.5pp):** Two negative results paragraphs (Deliverable 5), feature dominance analysis showing low JRN correlates with high own-table feature percentage (Table 9), honest limitations paragraph. The I-squared > 90% heterogeneity is addressed using Borenstein et al.'s [12] framework showing I-squared is a relative measure (proportion of variance due to true heterogeneity) not an absolute indicator of effect size variation, and recommending reporting of tau-squared and prediction intervals alongside I-squared.

**Section 7 (Conclusion, ~0.5pp):** Three future directions: GNN-based probes, temporal join extension, distributional JRN (JRN-variance, JRN-skewness).

## DELIVERABLE 3: Figure Plan (6 Figures)

Figure 1 (JRN Concept Diagram): Color-coded FK joins on the rel-f1 schema [14] showing hub nodes (results: 3 FKs) and bridge nodes (standings: 2 FKs) as identified by Chen et al. [4]. Green = high JRN (>1.3), yellow = critical zone (0.9-1.1), red = low JRN (<0.8). Small inset shows epidemiological R0 analogy.

Figure 2 (JRN Heatmap): Joins-by-tasks matrix across all 4 databases [14, 15, 16, 17] with diverging colormap centered at JRN = 1.0. Demonstrates both cross-join variance and within-database cross-task consistency.

Figure 3 (Probe vs Ground-Truth + Forest Plot): Panel A shows scatter of probe-estimated vs full-model JRN with points colored by dataset. Panel B shows meta-analytic forest plot with per-dataset Spearman rho, 95% CIs, pooled random-effects estimate, I-squared = ~92%, and prediction interval. This figure directly addresses the I-squared concern following Borenstein et al. [12].

Figure 4 (Cost-Efficiency Convergence): Two panels (rel-f1 [14] and rel-stack [15]) showing training curves for baseline, uniform, JRN-pruned, and JRN-guided architectures. Key visual: JRN-pruned and JRN-guided lines nearly overlap, demonstrating aggregation selection adds minimal benefit.

Figure 5 (Compounding Models): Scatter of predicted chain JRN (product of individual JRNs) vs measured chain JRN, with multiplicative and log-linear model fits and R-squared annotations. Points colored by chain length.

Figure 6 (Cross-Task Transfer Matrix): Four panels (one per database) showing Spearman correlation of JRN rankings between task pairs. High off-diagonal values (>0.7) demonstrate JRN captures structural rather than task-specific properties.

## DELIVERABLE 4: Table Plan (9 Tables)

Table 1 (Related Work comparison matrix): Method x Properties (Pre-training? Per-join? Threshold prediction? Architecture guidance? Heterogeneous graphs?) for ESNR [1], MPS-GNN [2, 3], RelGNN [4], DB Views [5], Relatron [6], Over-squashing [7, 8], and JRN.

Table 2 (Dataset statistics): Database x (#Tables, #FK Joins, #Rows, #Cols, Max Chain Depth, #Tasks, Metrics) for all four databases [14, 15, 16, 17].

Tables 3-9 cover JRN validation correlations, all JRN values, threshold negative result (monotonically increasing variance), compounding validation, main architecture results, cross-task transfer, and feature importance decomposition.

## DELIVERABLE 5: Negative Results Framing

Two paragraphs for the Discussion section, following Karl et al.'s [11] advice to emphasize novelty over performance and provide deep analysis of why negative results occur:

Paragraph 1 (Inverted-U Failure): Frames the disconfirmed threshold prediction as revealing a fundamental disanalogy between epidemic dynamics and relational information flow. In epidemics the transmission mechanism is fixed; in relational joins, different aggregation strategies change the nature of what is transmitted [9], not just the transmission rate.

Paragraph 2 (Architecture Guidance Weakness): Frames JRN's limitation as honest scope delimitation. JRN captures whether a join carries any signal (scalar property) but optimal aggregation depends on signal distribution across child entities (distributional property). JRN is a join selector, not an architecture configurator.

## DELIVERABLE 6: Reviewer Response Preparation (5 Rebuttals)

Objection A (no GNN experiments): Probes are deliberately cheap for pre-training use; validation shows correlation with full-model GNN join utility. Objection B (threshold fails): R0 analogy contributes at metric level (works) and intervention-impact level (fails); transparent reporting of both. Objection C (only 4 datasets): 4 of 7 RelBench v1 databases [10] covering diverse domains (motorsport [14], social Q&A [15], e-commerce [16], fashion retail [17]), scales (74K to 16.7M rows), and join topologies. Objection D (feature dominance): Low JRN correlating with high own-feature percentage is consistent with JRN's predictions. Objection E (high I-squared > 90%): Following Borenstein et al. [12], I-squared is relative, not absolute; all per-dataset correlations are positive.

## DELIVERABLE 7: Related Work Draft (6 Paragraphs)

Six comparison paragraphs positioning JRN against: (1) ESNR [1] — homogeneous graph noise metric vs per-join heterogeneous diagnostic; (2) MPS-GNN [2, 3] — during-training meta-path learning vs pre-training probes at less than 5% cost; (3) RelGNN [4] — blanket bridge/hub bypassing vs adaptive per-join allocation; (4) Database Views [5] — post-hoc explainability vs pre-training diagnostic; (5) Relatron [6] — whole-task RDL-vs-DFS selection vs per-join within-RDL configuration; (6) Over-squashing [7, 8] — full data graph curvature/commute-time analysis vs schema-level join-type measurement.

## DELIVERABLE 8: Contribution Bullets (5 Points)

Five bullet points: (1) JRN metric as first pre-training per-join diagnostic; (2) Empirical validation on 4 RelBench databases [10, 14, 15, 16, 17] spanning 24+ tasks; (3) Multiplicative compounding model for multi-hop chains; (4) Informative negative results disconfirming the inverted-U threshold prediction [11]; (5) Practical diagnostic with honest scope delimitation.

## Cross-Cutting Observations

Following Lipton's heuristics [13], contributions should appear by page 2, figures should tell the story independently, and reviewer concerns should be addressed proactively in the main text. Every known vulnerability (no GNN experiments, 4 datasets, high I-squared [12], feature dominance) has proactive mitigation text specified in the section-by-section plan.

## Sources

[1] [Towards Understanding and Reducing Graph Structural Noise for GNNs (ESNR) - Dong & Kluger, ICML 2023](https://proceedings.mlr.press/v202/dong23a.html) — Proposes edge signal-to-noise ratio (ESNR) metric using random matrix theory for evaluating graph structural noise; GPS graph rewiring method. Key difference from JRN: operates on homogeneous graphs, measures overall graph noise, not per-join transmission.

[2] [Meta-Path Learning for Multi-relational Graph Neural Networks - Ferrini et al., LoG 2024](https://arxiv.org/abs/2309.17113) — Proposes learnable meta-path scoring for multi-relational GNNs. Published in Proceedings of the Second Learning on Graphs Conference (LoG), PMLR 231:2:1-2:17, 2024. Key difference from JRN: learns join importance during training (expensive).

[3] [A Self-Explainable Heterogeneous GNN for Relational Deep Learning (MPS-GNN) - Ferrini et al., 2024](https://arxiv.org/abs/2412.00521) — Extends MP-GNN with learnable aggregate statistics (Meta-Path Statistics GNN). Supports scenarios where class membership depends on aggregate information from multiple meta-path occurrences. Self-explainable approach.

[4] [RelGNN: Composite Message Passing for Relational Deep Learning - Chen, Kanatsoulis, Leskovec, ICML 2025](https://arxiv.org/abs/2502.06784) — Introduces atomic routes and composite message passing to bypass bridge/hub nodes in relational graphs. SOTA on 30 RelBench tasks with up to 25% improvement. Published PMLR 267:8296-8312, 2025. Key difference from JRN: blanket strategy, not adaptive per-join.

[5] [Database Views as Explanations for Relational Deep Learning - Rissaki, Fountalis, Gatterbauer, Kimelfeld, 2025](https://arxiv.org/abs/2509.09482) — Post-hoc explainability via database view definitions identifying prediction-relevant subschemas using determinacy framework. Evaluated on RelBench. Key difference from JRN: post-hoc (requires trained model), not pre-training diagnostic.

[6] [Relatron: Automating Relational Machine Learning over Relational Databases - Chen et al., ICLR 2026](https://arxiv.org/abs/2602.22552) — Meta-selector between RDL and DFS using RDB-task homophily and affinity embeddings. Up to 18.5% improvement with 10x lower cost. Key difference from JRN: whole-task granularity, not per-join.

[7] [Understanding over-squashing and bottlenecks on graphs via curvature - Topping et al., ICLR 2022](https://arxiv.org/abs/2111.14522) — Introduces Balanced Forman curvature showing negatively curved edges cause over-squashing. ICLR 2022 Outstanding Paper Honorable Mention.

[8] [On Over-Squashing in Message Passing Neural Networks - Di Giovanni et al., ICML 2023](https://arxiv.org/abs/2302.02941) — Shows topology plays greatest role in over-squashing via commute time analysis. Width can help but depth cannot. Justifies graph rewiring. PMLR 202:7865-7885.

[9] [GNN-VPA: A Variance-Preserving Aggregation Strategy for Graph Neural Networks - Schneckenreiter et al., ICLR 2024 Tiny Papers](https://arxiv.org/abs/2403.04747) — Proposes variance-preserving aggregation using signal propagation theory. Shows sum aggregation causes exploding activations. Relevant for aggregation strategy variance discussion in JRN paper.

[10] [RelBench: A Benchmark for Deep Learning on Relational Databases - Robinson et al., NeurIPS 2024](https://arxiv.org/abs/2407.20060) — Comprehensive benchmark with 7 databases (v1), 11 databases (v2), 66 total tasks. Provides experimental infrastructure for all JRN experiments.

[11] [Position: Embracing Negative Results in Machine Learning - Karl et al., ICML 2024](https://arxiv.org/abs/2406.03980) — Argues for publishing negative results in ML. Key advice: emphasize novelty over performance, provide deep analysis of why negative results occur. Published PMLR v235:23256-23265.

[12] [Basics of meta-analysis: I-squared is not an absolute measure of heterogeneity - Borenstein et al., 2017](https://doi.org/10.1002/jrsm.1230) — Clarifies that I-squared is a relative measure (proportion of variance due to true heterogeneity) not an absolute measure of effect size variation. Recommends reporting tau-squared, tau, and prediction intervals.

[13] [Heuristics for Technical Scientific Writing (ML Perspective) - Zachary Lipton](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/) — Key advice: keep abstracts as 2-minute pitch, lead with problems not solutions, surface contributions by page 2-3, avoid intensifiers, anticipate reviewer questions proactively.

[14] [RelBench rel-f1 Dataset Page](https://relbench.stanford.edu/datasets/rel-f1/) — F1 database: 9 tables, 97,606 rows, 77 columns. 6 tasks. Hub nodes (results: 3 FKs) and bridge nodes (standings: 2 FKs) provide diverse join structures.

[15] [RelBench rel-stack Dataset Page](https://relbench.stanford.edu/datasets/rel-stack/) — Stack Overflow database: 7 tables, 4,247,264 rows, 52 columns. 6 tasks spanning classification (AUROC), regression (MAE), and link prediction (MAP).

[16] [RelBench rel-amazon Dataset Page](https://relbench.stanford.edu/datasets/rel-amazon/) — Amazon E-commerce database: 3 tables, 15,000,713 rows, 15 columns. 8 tasks. Simple star schema with max chain depth 2.

[17] [RelBench rel-hm Dataset Page](https://relbench.stanford.edu/datasets/rel-hm/) — H&M fashion database: 3 tables, 16,664,809 rows, 37 columns. 4 tasks. 7-day evaluation window provides temporal density contrast.

## Follow-up Questions

- Should the paper target a specific venue (NeurIPS, ICML, KDD), and how would the page limit and reviewer expectations differ across these?
- Should the paper include a RelBench v2 extension plan to preemptively address the 'only 4 datasets' concern, or would promising future work on all 11 databases suffice?
- Would a separate appendix with per-task JRN tables and full ablation results strengthen the submission, and what is the optimal balance between main paper clarity and appendix completeness?

---
*Generated by AI Inventor Pipeline*
