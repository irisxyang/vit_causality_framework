# Grounding Interpretability through Cross-Validation: A Causal Framework for Attribution in Vision Transformers

This repo contains the relevant code for _Grounding Interpretability through Cross-Validation: A Causal Framework for Attribution in Vision Transformers_ (2026).

The principle contribution of this thesis is methodological. We adapt two existing causal frameworks into a unified template for ViT interpretability: the activation-restoration frame-work of Meng et al. (2022) and the patch deletion framework of Petsiuk et al. (2018). Additionally, we develop a third intervention experiment, patch-level attention intervention.

Each experiment follows the same three-step structure with the same evaluation metric: 1) clean forward pass, 2) intervened forward pass, and 3) extract the change in true-class probability. This template allows the experiments to be comparable to each other, enabling cross-experiment analysis. The framework provides what existing post-hoc methods cannot: a measured causal grounding against which attribution claims can be validated rather than inferred.

The current code uses [Meta's DeiT-Tiny](https://arxiv.org/abs/2012.12877) and [ImageNet-1k-v1-enhanced](https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched), an annotated version of ImageNet-1k with detector-labeled subject bounding boxes. However, the framework provided is applicable to other vision transformers, other intervention targets, and other attribution methods.

## Abstract

Vision transformer interpretability has historically relied on post-hoc attribution methods that infer model behavior from a frozen network rather than measuring it causally. This leaves attribution claims unsubstantiated against the model’s actual computation. We address this gap by developing a causal evaluation framework for attribution method validation, organized around three intervention-based experiments adapted from existing causal frame-works: component recovery at multiple granularities, patch-level attention intervention, and patch deletion. Applying this framework to DeiT-Tiny on ImageNet-1k, we compare two widely used attribution methods, Integrated Gradients and Chefer relevancy, against the defined intervention-based methods. For our selected model (Meta’s DeiT-Tiny), our findings establish that Chefer recovers patch-level causal importance with high fidelity while Integrated Gradients fails on every metric we measured. We find that componential concentration in the hidden state systematically coexists with spatial redundancy in the input, producing a two- axis localization structure across classes. Finally, we establish that faithfulness and coherence in ViT attribution methods do not necessarily have a trade-off relationship. The framework is portable to other vision transformers and attribution methods, providing infrastructure for principled cross-validation of interpretability claims.

## Repo Guide

This repo contains a notebook file `vit_causality_framework.ipynb` with all experiment code, as well as `imagenet_classes.csv` with full list of ImageNet-1k classes. The code in `vit_causality_framework.ipynb` is structured with the following outline:

### Notebook Configuration

- Initial Dataset Load + Filter + Save
- Load Images from Save

This section initializes the data and variables used across this notebook. Insert your parent directory, HuggingFace token, and change other params as you see fit. If you have already initialized the dataset load once (and saved in a .pkl file in the specified directory), you can simply run the "Load Images from Save" cell.

### Visualization Functions

- Feature Inversion
- Representation Maximization
- Saliency Mapping

This section incldues implementations for all the visualization functions: Feature Inversion, Representation Maximization, and saliency mapping for attribution methods.

### Causal Tracing via Component Recovery

- Gaussian Noise Multiplier Tuning
- Activation-Level Component Recovery
  - Evaluation
- Channel-Level Component Recovery
  - Evaluation
  - Representation Maximization Visualization

This section implements component recovery at two granularities: activation-level and channel level. First, we tune the Gaussian noise multiplier for subject input patch noising. All subsequent component recovery experiments are performed with this parameter. For DeiT-Tiny on our 250-image subset of ImageNet-1k, we calibrate our sigma = 3.7331. Component recovery produces the most important components at the relevant granularity (blocks, sub-layers, channels) for carrying the class signal.

### Patch-Level Attention Intervention

- Attention Intervention Functions
- Identify Highest-Impact Block for Attention Intervention
  - Evaluation
- Full Dataset Run on Target Blocks
  - Evaluation

This section implements patch-level attention intervention, which is composed of suppression and concentration. This method is intended to evaluate the causality of attention weighting at subject versus non-subject patches throughout the model network. First, we perform a lightweight run across all transformer blocks to identify the highest-impact block for intervention. Then, we select the highest-impact blocks to evaluate intervention across the entire dataset.

### Attribution Method Comparison

- Integrated Gradients Implementation
- Chefer Saliency Implementation
- Comparison with IG and Chefer Saliency
  - Evaluation
- Attribution Method Evaluation via Patch Deletion Protocol
  - Evaluation
- Faithfulness-Coherence Analysis

In this section, we implement compare attribution methods through three separate experiments.

In the first experiment, we directly implement two existing attribution methods: Integrated Gradients (IG) by Sundararajan et al.,, and Chefer relevancy propagation by Chefer et al. We compare these two attribution methods with the causal tracing (component recovery) and attention intervention (suppression and concentration) implemented in previous sections.

In the second experiment we implement the patch deletion protocol from Petsiuk et al. (adapted for ViTs by Chefer et al.) as a causal benchmark for patch-level importance. Patch deletion validates causal contribution of patch saliency by iteratively removing top-k patches for increasing values
of k. As per the Petsiuk et al., we implement the AUC protocol for faithfulness evaluation. Under this evaluation, we also compare to three reference rankings: 1) a single-patch deletion “oracle,” which is the saliency ranking for each of 196 patches if that patch is deleted, 2) attention suppression at block 9 from the patch-level attention intervention, which is the saliency ranking based on suppressing block 9 attention at that patch, and 3) a random baseline, to ensure that the other attribution methods are actually producing meaningful results.

Finally, we evaluate the patch-level methods at in terms of faithfulness and coherence. Faithfulness refers to the AUC protocol from patch deletion, evaluating if the patches identified as high-importance are actually causally important to prediction. Coherence measures the interpretability of the final saliency mapping through three metrics: 1) Gini coefficient, which measures concentration (inequality) in a distribtuion, 2) bounding box Intersection over Union (IoU) at top 25% of patches, which measures what fraction of patches overlap with image bounding boxes of the actual subject, and 3) spatial contiguity, which computes the mean pairwise distance between the top 25% of patches as a measure of clustering.

## Save Directory Structure

The code will save experiment results at the specified `PARENT_DIR` in the following manner:

```text
PARENT_DIR/
├── inversion/ # all inversion visualizations
├── repmax/ # all repmax visualizations
├── saliency_panels/ # all saliency map comparison visualizations
│
|  # CAUSAL TRACING (COMPONENT RECOVERY) FILES
├── exp1_causal_tracing.pkl
├── sigma_calibration.pkl
├── exp1_distributed.png
├── exp1_flip_rate.png
├── exp1_headline.png
├── exp1_max_delta_by_class.png
├── exp1_max_delta_hist.png
├── exp1_peak_location.png
├── exp1_selected_images.png
├── exp1_single_image_heatmaps.png
│
├── exp1b_channel_ranking.png
├── exp1b_channel_tracing.pkl
├── exp1b_channels_to_visualize.pkl
├── exp1b_concentration.png
├── exp1b_max_per_image.png
│
|  # ATTENTION INTERVENTION FILES
├── exp2_block_curves.png
├── exp2_cross_exp_correlation.png
├── exp2_derived.pkl
├── exp2_full_sweep.pkl
├── exp2_lightweight_sweep.pkl
├── exp2_per_class_concentration_b4.png
├── exp2_per_class_scatter.png
├── exp2_per_class_suppression_b2.png
├── exp2_subj_vs_nonsubj.png
│
|  # IG AND CHEFER COMPARISON FILES
├── exp3_3a_correlation_hist.png
├── exp3_3b_subject_ratios.png
├── exp3_3b_topk_iou.png
├── exp3_attribution_sweep.pkl
├── exp3_per_class_scatter.png
├── exp3_token_group_fractions.png
│
|  # PATCH DELETION PROTOCOL FILES
├── exp4_componential_vs_spatial.png
├── exp4_deletion_by_regime.png
├── exp4_deletion_curves.png
├── exp4_deletion_sweep.pkl
├── exp4_oracle_rankings.pkl
│
|  # FAITHFULNESS-COHERENCE FILES
├── exp5_frontier.png
│
|  # FILTERED DATASET FOR EXPERIMENTS
└── experiment_images.pkl
```
