---
title: 'When Frozen ImageNet Features Lose: Transfer Learning on Galaxy Images'
date: 2026-07-16
permalink: /posts/2026/07/galaxy-morphology-transfer-learning/
tags:
  - deep-learning
  - pytorch
  - transfer-learning
  - computer-vision
  - class-imbalance
---

For image classification on a small dataset, the standard advice is to take a network pre-trained on ImageNet, freeze the backbone, and train a new classifier head. It is fast and needs almost no compute. On Galaxy10 it came last --- behind full fine-tuning, which I expected, and also behind a 0.31M-parameter CNN I trained from scratch, which I did not.

This post walks through the three-way comparison, why the frozen features did so badly, and what the rare classes look like when you report them honestly. The code is on [GitHub](https://github.com/ansingh16/galaxy_morphology_cnn).

## The data

[Galaxy10 SDSS](https://astronn.readthedocs.io/en/latest/galaxy10sdss.html) is 21,785 real SDSS galaxy cutouts, 69x69 RGB, hand-labelled into 10 morphology classes: round smooth galaxies, edge-on disks, barred and unbarred spirals, and so on.

It is heavily imbalanced. The largest class holds around 7,000 images. The smallest holds **17**. That ratio is roughly 411 to 1, and it drives nearly every modelling decision in the project.

Two consequences follow immediately:

- The loss is **class-weighted** by inverse frequency, so a mistake on the 17-image class costs far more than one on a 7,000-image class.
- Models are selected and reported on **macro-F1**, which averages the per-class F1 scores and so weights every class equally. A model that ignores the rare classes scores badly on it even when its accuracy looks respectable.

## Three approaches

All three train on the same stratified split and are judged on the same held-out test set.

| Approach | Trainable params | Test accuracy | Test macro-F1 |
|---|---|---|---|
| From-scratch CNN | ~0.31 M | 0.653 | 0.534 |
| ResNet-18, frozen features | ~5 K | 0.473 | 0.384 |
| ResNet-18, fine-tuned | ~11 M | **0.797** | **0.641** |

The from-scratch network is unremarkable by design: three conv-batchnorm-relu blocks with max pooling, adaptive average pooling, dropout, and a small classifier head. For the ResNet runs I swap the final layer for a 10-class head, resize the images to 96px, and normalise with ImageNet statistics instead of the galaxy channel statistics.

The only difference between the two ResNet rows is whether `requires_grad` is left on for the backbone.

## Why the frozen features lose

Freezing the backbone means the network describes each galaxy using filters learned from ImageNet, and only the 5K-parameter head gets to learn anything. Those filters are good at what ImageNet contains: edges of everyday objects, textures of fur and fabric, the parts of dogs and cars and furniture.

A 69px galaxy cutout has none of those. It is a small, faint, roughly symmetric smudge, and its class comes down to whether there is a bar through the middle, how tightly the spiral arms wrap, or whether the disk is edge-on. ImageNet's filters were never asked to describe that, and frozen, they cannot learn to. The head then has to separate 10 classes from a representation that has already discarded most of what distinguishes them.

The from-scratch CNN has far fewer parameters and no pre-training, but every filter it has was learned on galaxies. On this data that is worth more than a much larger set of filters learned on the wrong kind of image.

## Why fine-tuning still wins

Fine-tuning starts from the same ImageNet weights but lets the whole network keep training at a low learning rate. It lifts test macro-F1 from 0.534 (scratch) and 0.384 (frozen) to **0.641**, and accuracy to about 0.80, which is competitive with published Galaxy10 baselines.

So the pre-trained weights were not useless, they just were not usable as they came. Starting from them and then adapting them beat both training from scratch and taking them off the shelf.

This is why "use transfer learning" is too coarse an instruction to be much help. Freezing and fine-tuning are both transfer learning and they came 3rd and 1st here. The choice between them is the one that mattered, and the further your images sit from ImageNet, the more freezing costs you.

## Augmentation is free here, for a physical reason

Galaxies have no preferred orientation on the sky. Which way is up in a cutout is an accident of how the telescope happened to be pointing, so a spiral flipped left-to-right or rotated by any angle is still exactly the same kind of spiral.

That makes random flips and full 180-degree rotations **guaranteed** label-preserving, which is not something you usually get for free. Rotating a handwritten 6 gives you something that looks like a 9; rotating a barred spiral gives you a barred spiral. The augmentations apply to training only --- validation and test go through a plain resize-and-normalise pipeline, so evaluation stays deterministic and matches what the model sees at prediction time.

## The class the model cannot learn

Class 5, "Disk, Edge-on, Boxy Bulge", is the 17-image class. On the from-scratch model it scores an F1 of 0.167, by far the worst of any class and well under a third of the model's average.

Recall is not the interesting number here: the test set contains 2 images of this class, so recall is close to a coin flip either way. Precision is the one to look at, at 0.100. The model scatters boxy-bulge labels onto galaxies that are not boxy bulges. With 17 examples in the whole dataset it never learned what sets the class apart, so it guesses.

Fine-tuning improves things across the board but does not rescue this class, and its report says so too. A headline macro-F1 of 0.641 would not have told me any of this. The per-class table is what surfaces it, which is why the project reports one for every model.

## What I took away from this

First, the advice about frozen pre-trained features carries a domain assumption that usually goes unstated. It holds when your images look something like ImageNet's. Galaxy cutouts do not, and here the advice inverted.

Second, parameter count told me nothing useful. The frozen head has 5K trainable parameters and the fine-tuned network has 11M, but they run on an identical backbone. What separated them was which layers were allowed to keep learning, not how many numbers were involved.

Third, running the comparison was worth more than the winning number. If I had only trained the fine-tuned ResNet and reported 0.641, I would have had a result and no explanation for it. The frozen run is the one that told me why, and it is the run that lost.

The imbalance handling here follows the same reasoning I used on the [UK road accident severity classifier](/posts/2026/05/class-imbalance-uk-road-safety/) --- different data, same argument that accuracy is the wrong thing to optimise when one class dominates.
