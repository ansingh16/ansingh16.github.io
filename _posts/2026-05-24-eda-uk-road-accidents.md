---
title: 'Exploring 151K UK Road Accidents: An EDA Walkthrough'
date: 2026-05-24
permalink: /posts/2026/05/eda-uk-road-accidents/
tags:
  - eda
  - data-science
  - pandas
  - visualization
---

Before building any model, you need to understand the data well enough to make defensible modeling decisions. This post walks through how I approached exploratory data analysis on the UK Department for Transport's 2023 road accident dataset — 151,852 incidents across three linked files. The EDA directly shaped the dual-model classification strategy I described in my [class imbalance post](/posts/2026/05/class-imbalance-uk-road-safety/).

The project code is on [GitHub](https://github.com/ansingh16/UK_road_safety_modelling).

## The Data Sources

The DfT publishes three CSV files for each year's road accident statistics:

**Collisions** — one row per accident. Contains the where and when: latitude/longitude, date, time, local authority, road type, speed limit, junction detail, weather conditions, lighting, road surface. This is the anchor table.

**Vehicles** — one row per vehicle involved. Contains vehicle type (car, motorcycle, HGV, bicycle), manoeuvre at time of accident, junction location, skidding/overturning flags, first point of impact, and driver age/sex.

**Casualties** — one row per person injured. Contains casualty severity (slight, serious, fatal), casualty class (driver, passenger, pedestrian), age, sex, and pedestrian-specific fields (location, movement, direction).

A single collision can involve multiple vehicles and multiple casualties, so the relationship is one-to-many in both directions. The merge strategy matters.

## Loading and Merging

```python
import pandas as pd

collisions = pd.read_csv("dft-road-casualty-statistics-collision-2023.csv")
vehicles = pd.read_csv("dft-road-casualty-statistics-vehicle-2023.csv")
casualties = pd.read_csv("dft-road-casualty-statistics-casualty-2023.csv")
```

The tables link on `accident_index` (and additionally `vehicle_reference` between vehicles and casualties). For a collision-level classification, I aggregated vehicle and casualty features per accident — counts of vehicle types involved, worst casualty severity, total number of casualties — then merged back to the collisions table.

The key decision here: the target variable is the **worst casualty severity** in each collision, not the individual casualty outcomes. This gives one label per accident and avoids data leakage from casualty-level features that are only known after the fact.

## The Class Distribution Problem

The very first thing to check in any classification problem:

```python
collisions["accident_severity"].value_counts()
```

| Severity | Count   | Percentage |
|----------|---------|------------|
| Slight   | ~117,000 | 77%       |
| Serious  | ~32,000  | 21%       |
| Fatal    | ~2,100   | 1.4%      |

This immediately tells you that any model optimized for accuracy will get ~77% by predicting "Slight" for every case. The fatal class at 1.4% is practically invisible to a standard classifier. This single observation drove the entire modeling strategy — two separate models with different resampling approaches, detailed in the [companion post](/posts/2026/05/class-imbalance-uk-road-safety/).

## Feature Categories

The merged dataset contains a wide variety of feature types, each useful in different ways:

### Environmental Features
- **Weather**: fine, rain, snow, fog, high winds, other
- **Lighting**: daylight, darkness (lit/unlit), dawn/dusk
- **Road surface**: dry, wet, frost/ice, snow, flood

These encode the conditions at the time of the accident. Wet roads combined with darkness and high speed limits might correlate with more severe outcomes.

### Temporal Features
- **Date**: day, month, year — captures seasonal patterns
- **Time**: hour of day — rush hour vs late night
- **Day of week**: weekday commute patterns differ from weekend

Temporal features often carry strong signals. Late-night accidents may involve different driver demographics (fatigue, impairment) compared to morning commute collisions.

### Geographic Features
- **Latitude/Longitude**: exact accident location
- **Local authority**: administrative district
- **Road type**: motorway, A-road, B-road, minor road
- **Speed limit**: numeric (20, 30, 40, 50, 60, 70 mph)

Speed limit is particularly informative — higher limits correlate with higher-energy impacts and more severe outcomes. Road type and urban/rural classification provide context for the speed limit.

### Vehicle Features (Aggregated)
- Number of vehicles involved
- Types of vehicles (presence of HGVs, motorcycles, bicycles, pedestrians)
- Manoeuvre types (turning, overtaking, going ahead)
- Junction involvement

Collisions involving heavy goods vehicles or motorcycles tend toward more severe outcomes due to the mass differential and lack of protection respectively.

### Casualty Features (Aggregated, Non-Leaking)
- Total number of casualties per collision
- Whether pedestrians were involved
- Age distribution of casualties

Note: I deliberately excluded features that would only be known after the accident outcome (e.g., specific injury types) to avoid data leakage.

## Key EDA Observations That Shaped Modeling

Several patterns from the EDA directly influenced modeling decisions:

**1. Extreme class imbalance (1.4% fatal)** — This ruled out standard classification. Any approach needed explicit handling of the minority class, leading to the SMOTE+Tomek and ADASYN strategies.

**2. Speed limit as a strong predictor** — Higher speed limits showed clear association with severity. This feature was retained as-is (no binning needed) since LightGBM handles numeric features natively.

**3. Multi-vehicle accidents** — The number of vehicles involved and the types of vehicles are informative. Aggregating from the vehicles table to collision level preserved this signal without leaking individual vehicle outcomes.

**4. Temporal patterns** — Time of day and day of week showed non-uniform accident distributions. These were encoded as cyclical features (sine/cosine transforms) to preserve the circular nature of time.

**5. Geographic coverage** — The lat/lon data covers all of Great Britain. Rather than using raw coordinates (which would overfit to specific locations), geographic features were captured through road type, speed limit, and urban/rural classification.

## Practical EDA Checklist for Tabular Classification

Based on this project, here is the checklist I now use when starting any new tabular classification problem:

1. **Check class distribution first.** Before anything else. If the minority class is below 5%, plan for resampling or cost-sensitive learning from the start.

2. **Understand the entity relationships.** One-to-many relationships (one collision, many vehicles) require explicit aggregation decisions. Document your merge strategy early.

3. **Identify potential data leakage.** Any feature that encodes the outcome (or is only available after the outcome is determined) must be excluded. In accident data, injury details are leaky — they are the severity, not predictors of it.

4. **Profile feature types separately.** Environmental, temporal, geographic, and entity-specific features behave differently and may need different encoding strategies.

5. **Let the EDA inform the modeling strategy.** The 1.4% fatal rate was not a problem to solve in preprocessing — it was a fundamental characteristic of the domain that required a dual-model architecture. EDA should surface these structural properties before you write a single line of model code.
