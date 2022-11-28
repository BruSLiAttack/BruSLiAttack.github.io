## Welcome to GitHub Pages of BruSLiAttack

Reproduce our results: [GitHub](https://github.com/BruSLiAttack/BruSLiAttack.github.io)

Check out our paper: [BruSLiAttack: Bayesian Algorithm for Query-Efficient Score Based Sparse Attacks Against Black-Box Deep Learning Models](https://...)

Poster: [Poster](...)

#### ABSTRACT

The potential for extracting information, solely from the output of a machine learning model, poses safety and security threats against real-world systems; especially concerning in an era of proliferating models offered as Machine Learning as a Service (MLaaS). Sparse attacks are of particular interest. Because, they aim to discover the minimum number of perturbations to model inputs—l0 bounded perturbations—to craft adversarial examples to misguide model decisions, and expose a unique class of hidden model vulnerabilities. But, constructing sparse adversarial perturbations without prior model knowledge, against black-box models, even when models opt to serve confidence score information to queries—in a score-based attack setting—is non-trivial. Because, such an attack leads to: i) an NP-hard problem; and a ii) non-differentiable search space. We develop an algorithm built upon a Bayesian framework for the problem and evaluate against Convolutional deep Neural Networks (CNN), Vision Transformers (ViT) and recent Stylized ImageNet models (SIN). Importantly, vision transformers are yet to be investigated under a score-based attack setting. The attack demonstrate significantly high attack success rates with low query budgets; on the high-resolution ImageNet, 10 K queries achieve over 98% attack success rate against a CNN model, 87% against a ViT model and 97% against a SIN model in the hard setting of an attack to a target class with just 1% sparsity. Importantly, the highly query efficient algorithm we demonstrate to discover hidden model weaknesses raises new questions regarding the safety of deployed systems and the robustness of machine learning models.

#### ALGORITHM DIAGRAM
![Figure 1](image/Spa-diagram-horizon.pdf#gh-dark-mode-only)

Figure 1: An illustration of BruSLiAttack

#### VISUALIZATION

![Figure 2](image/visualization-GCV-fig1.pdf#gh-dark-mode-only)

Figure  2:  Targeted  Attack. Adversarial Example Created by BruSLiAttack to fool Google Cloud Vision.
