## Welcome to GitHub Pages of BruSLiAttack

Reproduce our results: [GitHub]([https://](https://github.com/BruSLiAttack/BruSLiAttack.github.io)) 

Check out our paper: [...](https://...)

Poster: [2022 Poster](...)

#### ABSTRACT

Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information for solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe.  Because, these attacks aim to minimize the number of perturbed pixels—measured byl0norm—required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting.  But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm—SparseEvo—for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based at-tack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks.  The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based Whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.

#### ALGORITHM DIAGRAM
![Figure 1](image/Spa-diagram-horizon.pdf#gh-dark-mode-only)

Figure 1: An illustration of BruSLiAttack

#### VISUALIZATION

![Figure 2](image/visualization-GCV-fig1.pdf#gh-dark-mode-only)

Figure  2:  Targeted  Attack. Adversarial Example Created by BruSLiAttack to fool Google Cloud Vision.
