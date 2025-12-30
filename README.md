# IG_for_bert
Implementing IG for BERT models : 

Reference : 
CAPTUM Library : 

https://captum.ai/tutorials/Bert_SQUAD_Interpret



Results : 

**Integrated Gradients**

Key implementation details:

[PAD] tokens are used as the baseline

[CLS] and [SEP] are masked only during visualization, not removed from the model

Masking is done to focus on semantic content tokens, since [CLS] and [SEP] are control tokens that often dominate attribution due to their structural role.

What the IG visualizations show

The IG plots highlight:

Positive (green) tokens → support the predicted class

Negative (red) tokens → oppose the predicted class

Neutral tokens → little influence

These attributions reflect causal sensitivity, not attention weights.

**Observations from IG**

**Fine-tuned BERT**

Attribution is distributed across meaningful predicates and rules. Both positive and negative evidence is visible. Conflicting facts receive different weights. This indicates: Evidence aggregation and task-aware reasoning.

**Base (pretrained) BERT**

Attribution is often more uniform. Repeated or frequent tokens dominate. Fewer strong negative attributions appear. This reflects: Generic lexical matching rather than structured reasoning.

IG convergence delta

The reported IG convergence delta measures numerical approximation quality:

The difference between the model’s output change and the sum of attributions.

**LayerConductance (Layer-wise Structural Analysis)**

While IG explains which tokens matter, LayerConductance explains:

At which transformer layers those tokens influence the final prediction. We focus specifically on the [SEP] token, which marks the boundary between segments (rules/facts vs query).

What LayerConductance measures

LayerConductance computes: How much a given layer’s activations contribute to the final output.

In this analysis: Conductance is computed per transformer layer. Attribution is extracted only for the [SEP] token. Values are normalized across layers. The result is a layer-wise contribution curve

This answers: How does reliance on [SEP] evolve across the network?

Results for Layer Conductance : 
**Fine-tuned BERT**

[SEP] contribution peaks in early–mid layers (≈ layers 3–5)

Contribution drops sharply after mid layers

Near-zero contribution in final layers

Interpretation:

The model uses [SEP] early to establish structure, then shifts to semantic reasoning for final decisions.

**Base (Pretrained) BERT**

[SEP] contribution also peaks in mid layers

However, contribution remains non-negligible into later layers

Slower decay toward zero

Interpretation:

The model continues to rely on structural boundary information close to the classifier.

**Together:**

IG shows semantic evidence aggregation
LayerConductance shows how fine-tuning reorganizes structural reliance. 
This multi-level analysis demonstrates that fine-tuning changes not only accuracy, but also the internal reasoning dynamics of the model.



Fine-tuning BERT shifts structural processing (e.g., [SEP]) to earlier layers and promotes content-driven reasoning in later layers, as evidenced by both token-level and layer-wise attribution analyses.This provides interpretable, causal evidence that the fine-tuned model reasons in a more task-aligned and faithful manner than the base pretrained model.


