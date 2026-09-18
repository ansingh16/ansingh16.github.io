---
title: 'The Model Was the Easy Part: A Laptop-Scale MLOps Pipeline'
date: 2026-09-18
permalink: /posts/2026/09/seoul-bike-mlops/
tags:
  - mlops
  - dvc
  - lightgbm
  - feature-engineering
  - docker
  - drift-monitoring
---

I spent an afternoon on a hyperparameter sweep for this model and it made things slightly worse. I added some weather and calendar interactions, and those hurt too. The thing that actually took the model from mediocre to good was a single plot of where it went wrong, and four features that followed from it. The model itself never changed.

That is the short version of a project I built to practise the engineering around a model rather than the model. It predicts hourly bike-rental demand in Seoul, but the prediction is almost beside the point. What I wanted was the pipeline: reproducible data preparation, schema validation, experiment tracking, a registry that will not let a worse model reach production, a containerised serving API, and drift monitoring wired up as a retrain trigger. All of it runs on a laptop with no cloud account. The code is on [GitHub](https://github.com/ansingh16/seoul-bike-mlops).

## The task, and a deliberately boring model

The data is the [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) set from UCI: 8,760 hourly records from December 2017 to November 2018, with weather and calendar fields. The job is to predict how many bikes get rented in a given hour.

The model is a LightGBM regressor with 600 trees, learning rate 0.03, 48 leaves, depth 8, trained on a log-transformed target because demand is right-skewed. That is the whole model. It lands here on a time-ordered holdout:

| Metric | Value |
|---|---|
| MAE | 189.6 |
| RMSE | 269.5 |
| R² | 0.797 |

I picked something small and unglamorous on purpose. A portfolio project built around a clever model tends to become a project about that one model, and the parts that actually decide whether a model is any use in practice never get built. I wanted those parts to be the content.

## Where the model was wrong, and why that mattered more than tuning

The first version of the model scored about 0.69 on the holdout. Okay, not good. The tempting next move is a bigger model or a hyperparameter search. Before doing either, it is worth asking *where* the predictions miss, because the answer often points at a missing feature rather than a missing parameter.

I plotted mean error by hour of day. The model systematically underpredicted the morning and evening commute peaks: at 8am and 6-7pm it forecast fewer bikes than actually went out. The weather-and-calendar features know that it is 6pm on a weekday, but they have no idea that the last few hours were already unusually busy. That recent level is exactly what the features were missing.

Demand is heavily autocorrelated. The strongest single predictor of how many bikes go out this hour is how many went out at the same hour yesterday, and the same hour last week. So I added four features: the count 24 hours ago, the count 168 hours ago, and the trailing 24-hour and 7-day means. Every one is built with `shift` of at least one step, so a row only ever sees hours that have already happened and the value being predicted can never leak into its own inputs.

That single change moved the holdout:

| | R² | RMSE |
|---|---|---|
| Weather + calendar only | ~0.69 | ~332 |
| With lagged demand | ~0.80 | ~270 |

RMSE dropped by roughly a fifth, and none of it came from a larger model. The hyperparameter sweep and the extra interactions I mentioned at the top were tried against this same holdout and both made it worse, so neither made the cut. The lags were the whole story. Error analysis was worth more than an afternoon of tuning, and it cost about ten minutes.

## The engineering around it

A good number on a holdout is a claim, not a result, until you can reproduce it and stop it from silently rotting. That is what the rest of the repo is for.

- **A DVC pipeline** runs the stages in order: `ingest`, `validate`, `featurize`, `train`, `evaluate`. `ingest` downloads the UCI data and splits it into a reference window for training and a later window held back for drift checks, so the whole thing rebuilds from source with one command.
- **Pandera** validates the data against a schema before any features are built, so a malformed pull fails the run instead of quietly producing a corrupted-but-plausible model.
- **MLflow** tracks every run's parameters, metrics, and model, backed by a local SQLite file so there is no server to stand up.
- **A gated registry** decides which version actually serves. A new model only takes the `champion` alias if it beats the incumbent on the holdout RMSE that was logged at training time; otherwise it is still registered, but the thing being served is left alone. That gate is the piece I care most about, and it has [its own write-up](/posts/2026/07/gated-model-registry-seoul-bike/), so I will not repeat it here.
- **A FastAPI service in a multi-stage Docker image** serves the chosen champion. The image copies in a self-contained `serving_model/` export and runs as a non-root user, with no MLflow store and no network needed at runtime.
- **Evidently** compares the reference and current windows and exits non-zero when too many features have drifted, so the check behaves like a retrain trigger rather than a dashboard nobody opens.

## The lag features came with a serving problem

There was a catch to the lag features, and it is the kind of thing that only shows up once you try to serve a model rather than just score it. "Is it 6pm on a weekend" can be computed from a single request. "How many bikes went out in the last 24 hours" cannot; it needs history the request does not carry.

So the model's inputs now split into two paths that have to end up identical:

- calendar and weather features, computed per row, the same way at training and serving time;
- the demand-history lags, computed in the batch pipeline during training, and supplied by the caller at serving time (in production, from an online feature store keyed on recent counts).

Both paths produce exactly the same columns, so there is still no train/serve skew. The lags are just supplied instead of derived, and the serving request grows four fields to carry them. Writing that constraint down explicitly, rather than hoping the serving code reconstructs the features the same way training did, is most of what keeps a served model honest.

## Closing the loop

The held-back window is entirely autumn, a season the model never trained on, which makes it a natural stress test. Running the monitor on it reports half the features drifted, past the configured threshold, and exits non-zero:

```
INFO 5 columns drifted (50% of monitored features) -> reports/drift.html
WARNING drift share 0.50 over threshold 0.30 -- retraining warranted
exit: 1
```

That exit code is the signal a scheduled job would retrain on. Drift says retrain, retraining produces a candidate, and the registry gate decides whether the candidate is allowed to serve. The three pieces only mean something together: automatic retraining without a promotion gate would just automate the risk of shipping a worse model.

## What I would keep

A few things from this project I would do again on the next one, roughly in the order they earned their place:

1. **Look at the errors before reaching for a bigger model.** A ten-minute plot of where the model was wrong beat an afternoon of tuning, and it is repeatable advice, not luck.
2. **Treat a holdout number as unfinished until it is reproducible.** The DVC and Pandera stages are unglamorous, but they are what let two runs differ because the model changed and not because the inputs drifted underneath them.
3. **Guard the one step that has consequences.** Registering a model version should be cheap; taking the serving alias should be the single gated decision.
4. **Design the features with serving in mind.** The lag features were a modelling win and a serving complication in the same move, and the complication is where train/serve skew hides.
5. **Make the monitor do something.** A drift report that only renders a page is easy to ignore. One that changes an exit code is a trigger.

None of this needed a cluster or a cloud bill. The point was never the R², and it was never the size of the data. It was building the scaffolding that turns a model into something you could actually run, retrain, and trust, at a scale small enough to hold in your head.
