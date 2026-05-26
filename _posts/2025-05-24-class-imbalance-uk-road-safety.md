---
title: 'Tackling Extreme Class Imbalance: A Dual-Strategy Approach for UK Road Accident Severity'
date: 2025-05-24
permalink: /posts/2025/05/class-imbalance-uk-road-safety/
tags:
  - machine-learning
  - class-imbalance
  - lightgbm
  - smote
---

When 98.6% of your labels belong to a single class, standard classifiers will happily predict the majority class every time and report impressive accuracy. This post walks through how I handled extreme class imbalance in a real dataset — 151,852 UK road accidents from 2023 — and why I ended up building two separate models instead of one.

The project code is on [GitHub](https://github.com/ansingh16/UK_road_safety_modelling).

## The Dataset

The UK Department for Transport publishes detailed records for every reported road accident. The 2023 release comprises three linked CSV files:

- **Collisions**: accident-level data — location (lat/lon), date and time, road type, speed limit, weather, lighting, road surface conditions
- **Vehicles**: vehicle-level data — type, manoeuvre, junction location, first point of impact
- **Casualties**: person-level data — severity of injury, age, sex, pedestrian movement

After merging and cleaning, the dataset contains 151,852 incidents distributed across three severity classes:

| Severity | Count | Percentage |
|----------|-------|------------|
| Slight   | ~117K | 77%        |
| Serious  | ~32K  | 21%        |
| Fatal    | ~2K   | 1.4%       |

Fatal accidents make up just 1.4% of the data. Any model trained on raw class proportions will learn to almost never predict "fatal" — which is exactly the wrong behavior if the goal is to catch life-threatening incidents.

## Why Two Models, Not One

Different stakeholders need different things from the same data:

**Emergency services** need to pre-position resources for accidents likely to be severe. For them, missing a fatal accident (false negative) is far worse than dispatching an extra ambulance that turns out to be unnecessary (false positive). The priority is **recall on the fatal/severe class**.

**Traffic management teams** allocate long-term resources — speed cameras, road redesigns, safety campaigns. They need a balanced view across all severity levels. Over-predicting severe cases wastes budget; under-predicting slight cases misses prevention opportunities. The priority is **balanced performance across all classes (macro recall)**.

Trying to optimize for both objectives in a single model means compromising on each. Instead, I built two separate LightGBM classifiers, each with its own resampling strategy and decision threshold.

## Resampling: SMOTE+Tomek vs ADASYN

Both models use LightGBM as the base classifier. The key difference is how the training data is resampled before fitting.

### SMOTE+Tomek (Emergency Response Model)

SMOTE (Synthetic Minority Over-sampling Technique) generates new synthetic samples for the minority class by interpolating between existing minority samples and their nearest neighbors. Tomek links then remove borderline majority-class samples that are nearest neighbors to minority samples, cleaning up the decision boundary.

The combination is aggressive: it both inflates the minority class and cleans ambiguous regions. This pushes the classifier toward higher sensitivity for the rare class, at the cost of more false positives in the majority class.

### ADASYN (Traffic Management Model)

ADASYN (Adaptive Synthetic Sampling) also generates synthetic minority samples, but it focuses generation on the minority samples that are hardest to classify — the ones surrounded by majority-class neighbors. This produces a more nuanced boundary that adapts to the local density of each class.

The result is a classifier that improves minority-class detection without being as aggressive as SMOTE+Tomek, leading to better balance across all classes.

### When to Choose Which

The choice between SMOTE+Tomek and ADASYN is not about one being universally better. It depends on what error is more costly:

- **High recall on rare class needed** (missing the event is catastrophic) → SMOTE+Tomek. Accept more false positives.
- **Balanced performance needed** (all classes matter roughly equally) → ADASYN. Accept slightly lower recall on the rarest class in exchange for fewer false alarms.

## Threshold Tuning via Probability Calibration

Resampling changes the class distribution that the model sees during training, which distorts the predicted probabilities. A model trained on SMOTE-resampled data will output inflated probabilities for the minority class because the training set made that class appear more common than it actually is.

To address this, I used probability calibration (Platt scaling) after training, then swept the decision threshold to find the operating point that best served each model's objective:

- **Emergency model**: threshold lowered to maximize severe-class recall
- **Traffic model**: threshold set to maximize macro recall across all three classes

This is a critical step that is often skipped. Without calibration, the threshold search operates on distorted probabilities and the selected operating point may not transfer well to production data.

## Results

| Model | Strategy | Severe Recall | Macro Recall |
|-------|----------|--------------|--------------|
| Emergency Response | SMOTE+Tomek | **92.4%** | Lower |
| Traffic Management | ADASYN | Lower | **81%** |

The emergency response model catches 92.4% of severe accidents — meaning fewer than 8 in 100 life-threatening incidents would be missed. The tradeoff is a higher false-positive rate on slight accidents, which is acceptable when the cost of missing a severe case is an under-resourced emergency response.

The traffic management model achieves 81% macro recall, distributing its predictive power more evenly. It is better suited for resource allocation decisions where over-reacting to slight accidents and under-reacting to severe ones are both undesirable.

## Key Takeaways

1. **Frame the problem before choosing the metric.** "Accuracy" is meaningless at 1.4% minority prevalence. Identify who will use the model and what errors cost them.

2. **Resampling is a modeling choice, not a preprocessing step.** SMOTE+Tomek and ADASYN encode different assumptions about which errors are acceptable. Choose based on the use case, not on which produces a higher number on a single metric.

3. **Calibrate before tuning thresholds.** Resampling distorts predicted probabilities. Threshold sweeps on uncalibrated scores will find suboptimal operating points.

4. **Two models can be better than one.** When stakeholders have genuinely different loss functions, a single model forces a compromise that serves no one well. Building separate models with separate resampling strategies and thresholds gives each stakeholder a tool optimized for their actual decision.

5. **SMOTE+Tomek is a reasonable default.** It improved minority-class recall in both model configurations. If you're unsure which resampling method to start with on an extreme-imbalance problem, it's a solid first choice.
