# Tuberculosis-Detection
Leveraging HPC with GAN-CNN Integration for Enhanced Tuberculosis Detection

Balancing the Imbalanced Dataset with GAN:
The project primarily solves the problem of unequal data in medical data.
Tuberculosis records (including chest X-rays) usually contain fewer
tuberculosis-positive samples compared to non-tuberculosis samples. To solve
this problem, we use generative adversarial networks to generate synthetic
data for composition. Create more realistic images. We leverage 4-8 GPU
nodes to do this work simultaneously.

CNN for Tuberculosis detection using HPC
The system uses convolutional neural networks (CNN) to classify chest
X-ray images into two groups: positive TB and negative TB. The CNN
architecture consists of three convolutional layers that progressively
extract features from the image. The number of filters increases from 16
in the first layer to 64 in the third layer, and each convolutional layer is
followed by a pooling layer to reduce dimensionality while preserving the
main features. The output method uses the sigmoid function to generate a
binary distribution (TB positive and TB negative). Use techniques such as
detraining techniques to avoid overload during training.

Communication and Synchronization Between Layers (Using InfiniBand)

Each convolutional layer processes its data and the results (feature maps) are
transferred to the next layer. In the case of using model parallelism, where
different layers of CNN are distributed across the nodes requires transferring
data between GPUs or nodes. This can introduce communication overhead.

HPC Enhancement: Use InfiniBand or other high-speed interconnects between
GPU nodes in order to reduce the latency involved in transferring data
between layers across different nodes

Gradient Synchronization Across Nodes

Averaging Gradients:

After each GPU node computes the gradients for its mini-batch of images,

all the gradients need to be synchronized across the nodes. This can be

done by collecting the gradient from all GPU and averaging them.

AllReduce or Ring AllReduce are the efficient wethods which can be

used to calculate the gradient aggregation in distributed environments

like our data Parallelism model.

How this works:

All the gradients from each of the GPU node are shared with other nodes.

Nodes compute average of the gradients shared in order to get a global

gradient which is used to update model weights
