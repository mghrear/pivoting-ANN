# pivoting-ANN
Adversarial Neural Network to create a classifier that is invariant with respect to vertex mass.

Base on the work in https://arxiv.org/abs/1611.01046
See also https://arxiv.org/abs/1703.03507, https://arxiv.org/abs/2011.08280

An adverserial training procedure is used to train a classifier that is a pivot with respect to invariant mass. This is useful for bump-hunts because it can search for a process while leaving the background distribution unchanged. This is also useful for domain adaption - because the classifier is a pivot with respect to invariant mass, it is expected generalize over different regions of invariant mass, allowing for robust training on side bands. For more details see https://docs.google.com/presentation/d/1x8XqCUZ9InPBTR0nRsNSBU2f1YJ3OmgUnJrwP-_dl5U/edit?usp=sharing 
