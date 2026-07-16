---
title: 'Forecasting M25 Congestion with an LSTM: The Baseline That Cheats'
date: 2026-07-16
permalink: /posts/2026/07/tfl-congestion-lstm/
tags:
  - deep-learning
  - lstm
  - time-series
  - pytorch
  - class-imbalance
---

One of the baselines I built for this project scores 0.975 accuracy and 0.948 F1, which comfortably beats my LSTM. It is also useless as a forecaster, and the reason it is useless is the most useful thing I got out of the project.

This post walks through the LSTM, the three baselines I measured it against, and why one of those baselines wins the table while telling you nothing. The code is on [GitHub](https://github.com/ansingh16/tfl-congestion-lstm).

## The task

Transport for London publishes sensor data for motorway checkpoints. Each one reports traffic volume and average speed every 15 minutes. I trained a PyTorch LSTM that reads a 4-hour window of recent history (16 intervals) and predicts whether the **next** interval will be congested, where congested means average speed below 30 mph.

The window's label is the congestion flag of the interval immediately *after* it, so no window ever contains its own answer. Training used a single busy checkpoint with a chronological 70/15/15 split --- chronological, not random, because shuffling time series across a split lets the model learn from the future to predict the past. Test results are on 15,731 held-out windows of 22 features each.

Congestion is the minority class, at roughly 10% of intervals across this site. That number is worth being careful with, though, and I was not careful enough with it at first.

## The class balance moves

Congestion is not spread evenly through the period, and the split is chronological, so each split sees a different balance:

| Split | Congested share |
|---|---|
| Train (first 70%) | 5.5% |
| Validation (next 15%) | 18.8% |
| **Test (last 15%)** | **25.6%** |

So the model trains on data that is 5.5% congested and is scored on data 4.7 times more congested than that. The site-wide ~10% describes the road accurately and describes the test set not at all, which matters because every number below was measured on the test set.

I had this wrong in my own writeups for a while, and the tell was sitting in the results the whole time: the majority-class baseline scores 0.744 accuracy, which only makes sense if the test set is 25.6% congested. If it were really 10%, that baseline would score around 0.90. Two numbers I had published side by side could not both be true.

The split is still right. The shift is a property of the road rather than a mistake in the code, and a random split would have hidden it by smearing the congested period evenly across train and test. But it does mean this is a distribution-shift problem as much as an imbalance problem, and the honest reading of the results has to say so.

## The results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **LSTM (one-step forecast)** | 0.846 | 0.628 | **0.977** | **0.765** |
| Majority class | 0.744 | 0.000 | 0.000 | 0.000 |
| Rush-hour rule | 0.740 | 0.490 | 0.344 | 0.404 |
| Speed threshold* | 0.975 | 1.000 | 0.901 | 0.948 |

The LSTM's average precision (PR-AUC) is **0.893**, against a no-skill baseline of 0.256.

## Three baselines, and what each one is for

Each baseline here answers a different question about the LSTM, which is why there are three of them rather than one.

**Majority class** predicts "not congested" every time. It scores 0.744 accuracy and 0.000 on precision, recall and F1. That single row is the imbalance argument: a model can be 74% accurate here while detecting no congestion at all. It also sets the floor every other accuracy number has to be read against, which makes the LSTM's 0.846 a more modest win than it looks in isolation --- and, as above, it is the row that tells you what the test prevalence actually is.

**Rush-hour rule** predicts congestion during fixed peak hours, and tests whether the LSTM has learned anything the clock does not already tell you. At 0.404 F1 the time of day clearly carries real signal, but a fixed schedule still catches only 34.4% of congestion --- traffic does not keep to the timetable.

**Speed threshold** is the one with the asterisk, and it needs a section of its own.

## The baseline that cheats

The speed-threshold rule predicts congestion from the current interval's average speed, which is the same quantity the label is defined from. Speed below 30 mph *is* the definition of congestion here, so the rule is reading the answer off the label definition one step late.

That 0.975 accuracy is really measuring how autocorrelated traffic speed is over 15 minutes. If the road is slow now it will probably still be slow in a quarter of an hour, so "congested now, therefore congested next" scores well without forecasting anything.

What separates it from the real competitors is information timing. The LSTM and the time-based rules only use information available **before** the interval they predict. The speed rule uses information from the interval it is scoring. So it belongs in the table as a check on my labelling rather than as a competitor, and that is what the asterisk says.

I kept it in for a practical reason. Anyone who works with time series will look at this problem and immediately ask "what about just thresholding current speed?" --- and if I had left it out, the obvious answer to that question would be that I never checked. It also does a job: it confirms the labels behave the way I think they do.

It is worth noticing how ordinary the cheat looks. It is not a bug, and nothing about the code says "leakage". It is just a feature that works suspiciously well, which is what most time-series leakage looks like from the inside.

## Reading the LSTM's numbers honestly

The LSTM catches 97.7% of congested intervals at 62.8% precision. Roughly a third of its congestion alerts are false alarms.

For this application I think that is a defensible trade. A missed jam means drivers routed into congestion they could have avoided; a false alarm means a slightly conservative route suggestion. The costs are not symmetric, so the operating point should not be either. That is a decision I made about the threshold, though, rather than something the model earned. The number that says the model genuinely ranks congestion well is the PR-AUC of 0.893 against a no-skill 0.256, because it does not depend on where I put the threshold.

The per-hour breakdown is less flattering. Accuracy and congested-class recall both dip around the evening peak, roughly 17--19h --- the window a clock-based rule handles worst, and the window where a forecast would be most worth having. So the model is at its weakest where it would be most useful. That belongs in the results rather than in a footnote.

## A note on what this project is not

This repo used to be described as reinforcement learning for traffic signal control, and it never was. There is no agent, no reward function, no policy and no environment --- it is a supervised LSTM doing binary classification on a forecasting task. I have since renamed the repository from `RL_signal_control` to `tfl-congestion-lstm` and fixed the description everywhere it appeared.

Worth saying plainly rather than letting the old label sit there: 0.977 recall on a minority class under a real distribution shift, measured against three named baselines including one built to catch my own labelling errors, is a better thing to have built than a vague gesture at RL would have been.

## Where this leaves it

The headline is 0.977 recall at 0.628 precision, and the honest reading of that is a model that will not miss much and will cry wolf about a third of the time it fires.

What I would take to the next project is mostly about the baselines. Picking ones that could actually beat me is what made the LSTM's numbers mean anything: the majority-class row is the only reason 0.846 accuracy is interpretable at all, and the speed-threshold row is what confirmed the labels were doing what I intended. A baseline you are guaranteed to beat is decoration.

The other habit worth keeping is checking *when* a model knew something rather than just what it knew. The speed rule has no bug in it and reads perfectly sensibly; it is wrong only in its timing. On a tabular problem I would be hunting for a leaky column, as I was with [the road accident severity data](/posts/2026/05/class-imbalance-uk-road-safety/), where a recoded severity field was sitting in the features. On time series the same mistake hides in the clock instead.
