**Federated Cifar10**

This repository contains Tensoflow code to generate (and save on
disk) a Federated version of Cifar10.

The Cifar10 dataset is partitioned following the paper [Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335): a Dirichlet distribution
is used to decide the per-client label distribution. 
A concentration parameter controls the identicalness among clients.
Very high values for `concentration` (e.g., > 100.0) imply an identical distribution of labels among clients,
while low (e.g., 1.0) values imply a very different amount of examples for each label in clients, and
for very low values (e.g., 0.1) all the client's examples belong to a single class.