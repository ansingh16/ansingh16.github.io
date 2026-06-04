---
title: 'Tackling Extreme Class Imbalance: UK Road Accident Severity with LightGBM'
date: 2026-05-24
permalink: /posts/2026/05/class-imbalance-uk-road-safety/
tags:
  - machine-learning
  - class-imbalance
  - lightgbm
  - feature-engineering
---

When 76% of your labels belong to a single class and the rarest class sits at 1.4%, standard classifiers will happily predict the majority class every time and report impressive accuracy. This post walks through what I learned building a severity classifier on 104,258 UK road collisions from the Department for Transport's 2023 STATS19 data, and why the numbers that looked great at first turned out to be completely wrong.

The project code and a live Streamlit dashboard are on [GitHub](https://github.com/ansingh16/UK_road_safety_modelling) and at [uk-road-safety-modelling.streamlit.app](https://uk-road-safety-modelling.streamlit.app/).

## The dataset

The DfT publishes three linked CSV files each year: collisions (one row per accident), vehicles (one row per vehicle involved), and casualties (one row per person injured). I used the collision and vehicle tables --- 104,258 collisions and 189,815 vehicle records.

| Severity | Count | Percentage |
|----------|-------|------------|
| Slight   | ~79K  | 76.1%      |
| Serious  | ~23K  | 22.5%      |
| Fatal/Severe | ~1.5K | 1.4%  |

A model that always predicts "Slight" gets 76% accuracy for free. Accuracy is the wrong metric here.

## The feature signal problem

Before worrying about class imbalance, I wanted to understand how much predictive power the features actually carry. I computed the mutual information between each numeric feature and the target.

The result was sobering. Average MI across all collision features was 0.011. For reference, anything above 0.05 is considered moderate signal. The strongest individual features --- `did_police_officer_attend`, `number_of_vehicles`, `speed_limit` --- barely reached 0.03. Most features were below 0.01.

This makes sense when you think about it. The collision table records the scene: road type, weather, lighting, speed limit, time of day. Whether someone dies in a crash depends on things like seatbelt use, exact impact angle, occupant age and frailty, vehicle safety rating --- none of which are in this dataset. We're trying to predict outcome from context, and context is a weak predictor.

## The leakage problem

My first models looked surprisingly good. A Random Forest with SMOTE+Tomek resampling was hitting 92% recall on severe cases. The feature importance plot told me why: a column called `enhanced_severity_collision` was dominating. A crosstab confirmed it mapped almost perfectly to the target --- it was a recoded version of `accident_severity` that the DfT includes for their own reporting purposes.

Once I dropped it, balanced accuracy fell from 84% to 55%. That was the honest baseline.

This is worth flagging because the column is not obviously leaky from its name. If you're working with the DfT STATS19 data, watch out for it.

I also looked hard at `did_police_officer_attend_scene_of_accident`, since 99% of severe collisions have police attendance. I kept it because police dispatch is based on the initial report (before severity is formally assessed), but it's borderline and worth noting.

## What actually helps: vehicle features and feature engineering

Since the collision features were weak, I looked at the vehicle table. Each collision involves 1.8 vehicles on average, so I aggregated to the collision level:

- **Vehicle type flags**: has_motorcycle, has_hgv, has_bicycle, has_pedestrian
- **Driver characteristics**: youngest driver age, oldest driver age, percent male drivers
- **Vehicle characteristics**: max engine capacity, max vehicle age
- **Crash dynamics**: any_skidding, any_side_impact

I also engineered features from existing columns:
- Parsed `time` into `hour`, created `is_night` (10pm--6am) and `is_weekend` flags
- Extracted `month` from `date`
- Replaced raw lat/lon with a coarser ~11km grid to reduce overfitting
- Dropped noise columns: `accident_year` (constant 2023), `local_authority_district` (all -1), IDs

The MI of the vehicle-derived features was comparable to the best collision features. Not transformative, but when every feature is weak, each small contribution matters. The final feature set has 42 features.

## Why two models, not one

Different stakeholders need different things:

**Emergency services** need to pre-position resources for potentially severe crashes. Missing a fatal accident is far worse than dispatching an extra ambulance. The priority is recall on the severe class.

**Traffic management** needs a balanced view for long-term resource allocation. Over-predicting severe cases wastes budget. The priority is balanced accuracy across all three classes.

## LightGBM with class weighting

I initially tried SMOTE, ADASYN, and SMOTE+Tomek resampling with Random Forest and Logistic Regression. After fixing the leakage, these produced mediocre results and were slow (36 model-resampling combinations, 20+ minutes to train).

LightGBM with the `class_weight` parameter turned out to be simpler and better. It handles imbalance natively during gradient computation --- no need to synthesize minority samples.

- **Severe-optimized model**: `class_weight={1: 50, 2: 3, 3: 1}` — heavy upweighting of the fatal/severe class
- **Balanced model**: `class_weight='balanced'` — inverse frequency weighting

Both use 500 trees, max depth 8, learning rate 0.03, and 63 leaves. Training takes under a minute.

## Threshold optimization

The default decision threshold (argmax of predicted probabilities) is conservative. For the severe-optimized model, sweeping the probability threshold for class 1 trades accuracy for recall:

| Threshold | Severe Recall | Macro Recall | Accuracy |
|-----------|--------------|--------------|----------|
| 0.01      | 99.7%        | 0.425        | 13%      |
| 0.03      | 97.4%        | 0.493        | 28%      |
| 0.05      | 93.4%        | 0.493        | 37.6%    |
| default   | 34.2%        | 0.527        | 66.8%    |

This is the most useful output of the project. Instead of a single accuracy number, the model provides a curve that lets the user pick their tradeoff. The Streamlit dashboard exposes this directly.

## How this compares to published work

I looked for peer-reviewed studies on the same DfT STATS19 dataset to see if anyone had done significantly better.

[Le (2026)](https://doi.org/10.1371/journal.pone.0347873) in PLOS ONE used LightGBM with SMOTE on 503K STATS19 records (2020--2024) for 2-class KSI prediction. At an optimized threshold, KSI recall reached 0.605 with ROC-AUC of 0.664. Our 3-class macro recall of 0.527 is broadly consistent given the harder multiclass setting.

[Lagias et al. (2022)](https://doi.org/10.1007/978-3-031-08223-8_34) at Springer EANN explicitly framed STATS19 severity prediction as an imbalanced-classification benchmark, noting ~50% missing data and extreme class skew. Their ANN and RL baselines produced modest results.

[Obasi & Benson (2023)](https://doi.org/10.1016/j.heliyon.2023.e18812) in Heliyon report 87% overall accuracy with Random Forest on STATS19 data from 2005--2014. Without per-class recall breakdown, that 87% mostly reflects correctly predicting the ~80% Slight majority.

No published study achieves strong 3-class severity prediction from pre-crash features alone.

## Final results

| Model | Severe Recall | Macro Recall | Accuracy |
|-------|--------------|--------------|----------|
| Severe-optimized (LightGBM) | 0.342 | **0.527** | 0.668 |
| Balanced (LightGBM) | 0.316 | 0.520 | 0.647 |
| Baseline (always Slight) | 0.000 | 0.333 | 0.761 |

These numbers are honest. They reflect a genuinely hard classification problem where scene-level features have limited predictive power for crash severity. The value is not in the accuracy number but in making the tradeoff explicit and adjustable.

## What I took away from this

First, always look at the feature importance plot before celebrating good metrics. The leakage column was sitting in plain sight.

Second, overall accuracy is worse than useless on imbalanced problems --- it actively misleads. A 76% accurate model that misses every severe crash is harmful if deployed.

Third, when individual features are all weak (MI < 0.02), the approach matters less than the features. SMOTE vs class weighting vs threshold tuning are all rearranging deck chairs if the features don't carry signal. The vehicle table aggregation added more value than any resampling strategy.

Fourth, sometimes the most honest thing a model can do is show you the tradeoff curve and let you pick your operating point. Not every problem has a clean solution.
