---
title: 'Exploring UK Road Accidents: What 104K Collisions Tell You Before You Model'
date: 2026-05-24
permalink: /posts/2026/05/eda-uk-road-accidents/
tags:
  - eda
  - data-science
  - pandas
  - visualization
---

Before building any model, you need to understand the data well enough to make defensible modeling decisions. This post walks through how I approached exploratory data analysis on the UK Department for Transport's 2023 road accident dataset --- 104,258 collisions and 189,815 vehicle records. The EDA directly shaped the dual-model strategy I describe in my [class imbalance post](/posts/2026/05/class-imbalance-uk-road-safety/).

The full analysis is in the [feature analysis notebook](https://github.com/ansingh16/UK_road_safety_modelling/blob/main/notebooks/03_Feature_Analysis.ipynb). The project code is on [GitHub](https://github.com/ansingh16/UK_road_safety_modelling).

## The data sources

The DfT publishes three CSV files each year as part of their STATS19 road casualty statistics:

**Collisions** --- one row per accident. Contains the where and when: latitude/longitude, date, time, local authority, road type, speed limit, junction detail, weather conditions, lighting, road surface. This is the anchor table.

**Vehicles** --- one row per vehicle involved. Contains vehicle type (car, motorcycle, HGV, bicycle), manoeuvre at time of accident, skidding/overturning flags, first point of impact, engine capacity, and driver age/sex. About 1.8 vehicles per collision on average.

**Casualties** --- one row per person injured. Contains casualty severity, class (driver, passenger, pedestrian), age, sex. I did not use this table for modeling --- casualty details are essentially the outcome, and including them would leak the target.

## Loading and merging

The collision table is the base. The vehicle table needs aggregation because multiple vehicles map to a single collision. I built collision-level summaries:

```python
vehicle_agg = vdf.groupby('accident_index').agg(
    has_motorcycle=('vehicle_type', lambda x: int(any(x.isin([2, 3, 4, 5])))),
    has_hgv=('vehicle_type', lambda x: int(any(x.isin([19, 20, 21])))),
    has_bicycle=('vehicle_type', lambda x: int(any(x == 1))),
    driver_age_min=('age_of_driver', lambda x: x[x > 0].min()),
    engine_cc_max=('engine_capacity_cc', lambda x: x[x > 0].max()),
    any_skidding=('skidding_and_overturning', lambda x: int(any(x.isin([1,2,3,4,5])))),
    any_side_impact=('first_point_of_impact', lambda x: int(any(x.isin([3, 4])))),
    # ... plus pct_male_drivers, driver_age_max, has_pedestrian, age_of_vehicle_max
).reset_index()
```

This preserves what matters --- whether vulnerable road users were involved, the range of driver experience, vehicle characteristics --- without leaking individual vehicle outcomes. The merge is a left join on `accident_index`, keeping every collision.

## The class distribution

The first thing to check:

| Severity | Count | Percentage |
|----------|-------|------------|
| Slight   | ~79K  | 76.1%      |
| Serious  | ~23K  | 22.5%      |
| Fatal/Severe | ~1.5K | 1.4%  |

The imbalance ratio (Slight to Fatal) is 52:1. Any model optimized for accuracy will get 76% by predicting "Slight" for every case. This single observation means accuracy is the wrong metric.

## How much signal do the features carry?

This is where the analysis got interesting. I computed mutual information between each feature and the target. MI measures statistical dependence regardless of whether it's linear or nonlinear.

The average MI across collision features was **0.011**. For context, MI above 0.05 is moderate. The strongest features --- `did_police_officer_attend` (MI ~0.03), `number_of_vehicles`, `speed_limit` --- barely cracked 0.02.

The vehicle-derived features (`engine_cc_max`, `has_motorcycle`, `any_side_impact`) had MI in the same range as the best collision features. Not a dramatic improvement, but meaningful when everything is weak.

This tells you something fundamental: the features we have describe the scene (road, weather, time, location), not the crash dynamics (impact angle, seatbelt use, vehicle safety rating). Scene conditions are weak predictors of injury severity. This is a hard problem not because of the imbalance alone, but because the features genuinely don't discriminate well.

## Feature engineering

A few transforms helped extract more signal from existing columns:

**Time parsing**: The raw `time` column is a string like "14:30". Parsing it into `hour` and then creating an `is_night` flag (10pm--6am) and `is_weekend` flag captures the temporal patterns that matter --- late-night accidents involve different driver demographics than morning commutes.

**Date to month**: Seasonal patterns in accident severity. Winter months, shorter days.

**Geo-grid**: Raw latitude and longitude overfit to specific locations. Rounding to a ~11km grid captures regional effects (urban density, road network quality) without memorizing individual intersections.

**Noise removal**: `accident_year` is constant (all 2023), `local_authority_district` is all -1 (missing), and `accident_index`/`accident_reference` are just IDs. Dropping these prevents the model from fitting noise.

The final feature set has 42 features: 25 from the collision table, 11 vehicle aggregates, and 6 engineered.

## The leakage discovery

A column called `enhanced_severity_collision` showed up as the top feature by a wide margin. A crosstab against the target showed near-perfect correspondence --- it's a DfT administrative recoding of `accident_severity`. Including it inflated balanced accuracy from 55% to 84%.

This is easy to miss because the column name doesn't scream "I'm the target in disguise." If you're working with STATS19 data, drop it.

I also investigated `did_police_officer_attend_scene_of_accident`. Police attend 99% of fatal collisions but only 63% of slight ones. It's borderline: police are dispatched based on initial reports, before severity is formally assessed. I kept it but flagged the caveat in the analysis.

## Key observations that shaped modeling

**1. Extreme imbalance (52:1)** required models that explicitly handle class imbalance. I ended up using LightGBM's `class_weight` parameter rather than SMOTE-style resampling, since it handles imbalance during gradient computation without fabricating synthetic samples.

**2. Weak individual features** (avg MI = 0.011) meant that the model architecture matters less than the features. Moving from Random Forest to LightGBM improved accuracy from ~55% to ~65%, but adding the vehicle table features contributed just as much.

**3. Two stakeholders, two tradeoffs.** Emergency services want maximum severe recall (catch every bad crash, accept false alarms). Traffic management wants balanced accuracy. This led to two models with different class weight schemes, plus threshold optimization to let the user pick their operating point.

**4. The vehicle table is underused.** Most projects I've seen on this dataset only use the collision table. The vehicle-level data --- especially motorcycle/HGV flags, driver age, and engine capacity --- adds features with MI comparable to the best collision features.

**5. Geographic features are tricky.** Raw lat/lon overfits. The geo-grid approach (rounding to ~11km cells) was a reasonable middle ground, but there might be better approaches using regional socioeconomic data or road network features.

## Practical EDA checklist for tabular classification

Based on this project:

1. **Check class distribution first.** Before anything else. If the minority class is below 5%, plan for cost-sensitive learning from the start.

2. **Compute mutual information early.** It takes one function call and immediately tells you whether you have strong features or weak ones. This determines how much value you'll get from model tuning vs feature engineering.

3. **Understand entity relationships.** One-to-many relationships (one collision, many vehicles) require explicit aggregation decisions. Document your merge strategy and justify what you aggregate vs drop.

4. **Hunt for leakage before modeling.** Any feature that encodes or is derived from the target must be excluded. Check feature importance on a quick baseline model and investigate anything suspiciously strong.

5. **Let the EDA determine your expectations.** With average MI of 0.011, I should have expected 60--70% accuracy at best. Understanding this early prevents wasted effort chasing unrealistic targets and helps you frame the results honestly.
