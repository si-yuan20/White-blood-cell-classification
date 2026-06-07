# Theoretical complexity notes

This document provides theoretical complexity analysis corresponding to Table 13 of the paper:

**Edge-prior and reliability-guided collaborative learning for white blood cell classification**

---

## 1. ConvNeXt branch

**Main operations:**
- 7x7 depthwise convolution (stem)
- 1x1 pointwise convolutions
- Layer normalization
- GELU activations

**Dominant term:** Depthwise convolutions.

For an input of size H×W×C_in:
- Depthwise conv (kernel K): O(H · W · C_in · K²)
- Pointwise conv: O(H · W · C_in · C_out)

The ConvNeXt base architecture has ~89M parameters (pre-trained backbone).

Big-O for ConvNeXt branch: **O(H · W · C · K²)** where K is kernel size and C the maximal channel count.

---

## 2. Mamba branch

**Main operations:**
- Patch embedding (conv2d, stride=patch_size)
- 4× Mamba blocks (SSM-based sequence modeling)

For an input of size H×W:
- Patchify: O(H · W · C_in · P² / P²) = O(H · W · C_in) (stride = P)
- Tokens: N = (H/P) · (W/P), each of dimension D = 512
- Each Mamba block: approximately **O(N · D²)** for the SSM and projection layers

The SSM core has linear complexity in sequence length N (unlike Transformer's O(N²)),
so the **Mamba branch approximately scales linearly with token length** for the state-space
sequence modeling component. The projection layers dominate with O(N · D²).

Big-O for Mamba branch: **O(N · D²)** where N = (H/P)·(W/P) and D is the embedding dimension.

---

## 3. SAF module (Structure-Aided Attention Fusion)

**Components:**

### 3.1 Edge prior extraction
- Sobel filtering: O(H · W)

### 3.2 Feature projection
- 1x1 conv for ConvNeXt features: O(H' · W' · C_conv · D_saf)
- 1x1 conv for Mamba features: O(H' · W' · C_mamba · D_saf)

### 3.3 Bidirectional cross-attention
- Tokenize: O(H' · W' · D_saf)
- Two MHA blocks with N_tokens = T² (e.g., T=14):
  O(T² · D_saf²) per attention head × num_heads

The cross-attention scales quadratically with the number of spatial tokens.

### 3.4 Edge-context aggregation
- Multi-scale pyramid (3 scales): O(H' · W' · D_edge) per scale

### 3.5 Depthwise separable fusion
- Depthwise conv: O(H' · W' · D_saf · K²)
- Pointwise conv: O(H' · W' · D_saf²)

**SAF overall:** While the Mamba branch has linear sequence complexity, SAF introduces
**attention-related O(T²) terms** through bidirectional cross-attention. The full framework
is **not purely linear** in complexity.

---

## 4. Reliability-guided bilateral fusion (RGBF)

**Components:**

### 4.1 Branch logits
- Two linear projections: O(C_conv × num_classes) + O(C_mamba × num_classes)

### 4.2 Reliability gate
- Softmax, entropy, margin: O(num_classes) per branch
- 2-layer MLP with hidden dim 128: O(10 × 128 + 128 × 128 + 128 × 2)
- Constant complexity per sample: **O(1)** relative to image size

### 4.3 Weighted fusion
- O(2 × D_rgbf) for weighted sum and final classification

**RGBF overall: O(1)** per sample — negligible relative to backbone costs.

---

## 5. Classification head

- Linear projection: O(D_fuse × num_classes)
- Negligible.

---

## 6. Full framework complexity summary

| Component                | Big-O                           |
| ------------------------ | ------------------------------- |
| ConvNeXt backbone        | O(H · W · C_max · K²)          |
| Mamba encoder            | O(N · D²)  (N = 196, D = 512)  |
| SAF (cross-attention)    | O(T² · D_saf²)  (T = 14)       |
| SAF (conv layers)        | O(H' · W' · D_saf² · K²)       |
| RGBF gate                | O(1) per sample                 |
| Classifier               | O(D_fuse × num_classes)         |

**Total:** Dominated by ConvNeXt backbone convolutions and SAF attention operations.
The Mamba SSM core has linear sequence complexity, but SAF's bidirectional
cross-attention introduces a quadratic term relative to token grid size (T=14).

For a 224×224 input with patch_size=16:
- ConvNeXt features: 14×14 spatial grid
- Mamba tokens: N = 196
- SAF tokens: T² = 196 (default token_hw=14)

---

## 7. Practical complexity (Table 12 companion)

Beyond theoretical Big-O, Table 12 in the paper reports empirically measured complexity:

- **Parameters (M):** Total and trainable parameter counts
- **FLOPs (G):** Measured via thop (MACs × 2 approximation)
- **Inference time (ms):** Mean latency over 200 forward passes (30 warmup) on RTX 3090

Note: FLOPs for Mamba-based models via thop may not perfectly capture SSM operations,
which involve recurrent state computations that toolkits like thop do not fully model.
The reported FLOPs should be interpreted as approximate.
