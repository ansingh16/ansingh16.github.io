---
title: 'A Gated Model Registry: How to Stop a Worse Model Reaching Production'
date: 2026-07-16
permalink: /posts/2026/07/gated-model-registry-seoul-bike/
tags:
  - mlops
  - mlflow
  - model-registry
  - dvc
  - lightgbm
---

There is a failure mode that most deployment tutorials skip past: you retrain, the new model is worse than the one you already have, and it replaces it anyway. Nothing errors. Nothing looks wrong in the logs. The predictions just get worse, and you find out later than you would like.

This post walks through the promotion gate I built on the model registry to prevent that, and why comparing two RMSE values honestly turned out to depend on a lot of unglamorous pipeline work underneath. The code is on [GitHub](https://github.com/ansingh16/seoul-bike-mlops).

## The project, briefly

The model predicts how many bikes get rented per hour in Seoul from weather and calendar features, using the [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) dataset from UCI: 8,760 hourly records spanning December 2017 to November 2018.

The model is deliberately small and boring. It is a LightGBM regressor, 600 trees, learning rate 0.03, 48 leaves, max depth 8, trained on a log-transformed target. It lands at:

| Metric | Value |
|---|---|
| MAE | 189.6 |
| RMSE | 269.5 |
| R² | 0.797 |

That R² of 0.797 is not what the repo is for. Everything around the model is: a reproducible DVC pipeline (ingest, validate, featurize, train, evaluate), Pandera schema validation, MLflow tracking on a SQLite backend, a gated registry, a FastAPI service in a multi-stage Docker image, and Evidently drift monitoring. All of it runs locally on a laptop with no cloud dependencies.

## The failure the gate prevents

A registry that only records model versions does not protect you from anything on its own. The question that matters is which version is *serving*, and if the answer is always "the newest one", then every retrain is an unreviewed production change.

Retrains get worse for ordinary reasons. A data pull silently returns a partial month. A feature changes meaning upstream. A hyperparameter edit that helped on one split hurts on another. None of these announce themselves. They just produce a model that trains successfully and scores worse.

So the rule I implemented is that a new version only takes the `champion` alias if it **beats the incumbent on the holdout RMSE logged at training time**. Otherwise it is still registered --- the run is tracked, the artifact is kept, you can go look at it --- but the alias does not move, and the thing being served is untouched.

## What it looks like when it runs

The first run has no incumbent, so it promotes unconditionally:

```
INFO registered seoul-bike-demand v1 (rmse=331.7)
INFO no incumbent -- v1 is the first champion
```

Later I added lag features and retrained. That model was genuinely better, so the alias moved:

```
INFO promoted v5: 269.5 beats champion v1 331.7
INFO exported champion v5 to serving_model/
```

And this is the case the whole thing exists for --- a retrain that did not improve:

```
INFO kept champion v5 (269.5); v6 did not improve (274.1)
```

v6 is still a real, registered model version that I can go and inspect. It just is not serving. Without the gate it would have replaced the 269.5 model, and the only symptom would have been predictions that were slightly worse than the week before.

## Comparison has to be like-for-like

The gate is only honest if the two RMSE values mean the same thing. That is a real constraint on the rest of the pipeline, and it is why the boring parts matter:

- The metric is logged **at training time**, against a fixed holdout fraction, by the training code itself. It is not recomputed later by the registry against whatever data happens to be lying around.
- The DVC pipeline makes the data-preparation stages reproducible, so two runs differ because the model differs, not because the inputs drifted underneath them.
- Pandera validates the data before featurization, so a schema violation fails the run instead of producing a model with a corrupted-but-plausible score.

A promotion gate on top of an unreproducible pipeline would just be comparing two numbers that were never comparable in the first place, and doing it with more ceremony. The DVC and Pandera stages are what let the comparison mean something.

## Serving the champion, not the newest

Promotion also re-exports the current champion to a self-contained `serving_model/` directory, along with an `info.json`:

```json
{
  "name": "seoul-bike-demand",
  "alias": "champion",
  "version": 5,
  "log_target": true,
  "rmse": 269.5260217436225
}
```

That file is what the Docker image copies in and the FastAPI app loads, so the service is pinned to a specific chosen version rather than to whatever trained last. It also carries the `log_target` flag, which matters more than it looks: the model predicts a log-transformed target, and serving code that forgets to invert the transform returns wrong numbers without raising anything.

That is train/serve skew, and the fix rhymes with the gate --- write the decision down and ship it with the artifact instead of hoping the serving code remembers it.

## Closing the loop with drift

The last piece is Evidently. The data is split at 2018-09-01 into a reference period and a current period, and the monitoring step compares them. If drift exceeds the configured threshold the check exits non-zero, so it acts as a retrain trigger instead of a dashboard nobody opens.

Those two pieces work together. Drift says retrain, retraining produces a candidate, and the gate decides whether the candidate is allowed to serve. Wiring up automatic retraining without a promotion rule would just automate the failure I started this post with.

## If you're adding a registry to your own project

A few things I would do again, in the order they mattered:

1. **Decide what "better" means before you need it.** Here it is holdout RMSE, lower wins, logged by the training code. Picking the metric while looking at a candidate's score is how gates get talked around.

2. **Log the comparison metric at training time, not promotion time.** If the registry recomputes the score later against whatever data is lying around, you are comparing two different experiments and calling it a gate.

3. **Make the alias the only thing that matters.** Registering a version should be cheap and consequence-free; taking the `champion` alias should be the one guarded step. That way there is no pressure to avoid registering a bad run, and you keep the record.

4. **Ship the serving decisions with the artifact.** `info.json` carries the version and the `log_target` flag into the container. Anything the serving code has to remember on its own, it will eventually forget.

5. **Check the gate rejects.** Everyone tests the promote path because it is the happy one. `kept champion v5 (269.5); v6 did not improve (274.1)` is the line that told me the thing actually worked, and it is the line I had to go out of my way to produce.
