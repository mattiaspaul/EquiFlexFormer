# EquiFlexFormer
MELBA 2025 submission
## Look, No Convs! Equivariance for Vision Transformers in Medical Image Segmentation


## ✅ Project Checklist

- [x] Initial commit
- [x] Implementation of Flex-Attention
- [x] Implementation of Equivariant Network
- [ ] Comparison with CNNs
- [ ] Upload trained models

## Abstract 

While medical image segmentation has achieved great success using convolutional neural networks, particularly U-Net architectures, its practical performance and robustness still heavily rely on ad hoc post-processing strategies. Test-time augmentation, for example, is often employed to achieve equivariance to mirrored, rotated, or permuted inputs.

In contrast, vision transformers could, in principle, be less susceptible to such issues, as the self-attention mechanism is inherently equivariant to permutations of the input. However, their practical performance in medical image segmentation remains limited, primarily due to their need for large amounts of training data, given the absence of the inductive biases found in CNNs. Additionally, operations like patch embedding and positional encoding break the permutation equivariance of vision transformers, and large patch sizes tend to degrade segmentation quality.

In this work, we propose a new approach that enables equivariance to geometric transformations, such as reflections, in the early layers of a vision transformer while preserving strong model performance. By employing robust self-supervised pretraining and small patch sizes, we achieve state-of-the-art performance on several 2D segmentation benchmarks. Our method outperforms ResNet models pretrained with supervision and certain types of equivariant U-Nets, and performs comparably to more complex variants with test-time augmentation.

Our code is available at https://github.com/mattiaspaul/EquiFlexFormer.


