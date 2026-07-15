---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

<h2 class="section-heading"><span class="section-heading__accent">//</span> GenAI &amp; NLP</h2>

<div class="projects-grid">

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-robot"></i></div>
      <div class="project-card__category">RAG / NLP</div>
    </div>
    <h3 class="project-card__title">UK Visa RAG Chatbot</h3>
    <p class="project-card__desc">RAG chatbot that ingests UK Immigration Rules via GOV.UK API, embeds them with sentence-transformers (all-MiniLM-L6-v2), stores in FAISS/ChromaDB, and generates answers with source attribution through a Streamlit chat interface.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">RAG</span>
      <span class="project-card__tag">FAISS</span>
      <span class="project-card__tag">ChromaDB</span>
      <span class="project-card__tag">Streamlit</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/UK_visa_RAG" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-file-alt"></i></div>
      <div class="project-card__category">LLM Engineering</div>
    </div>
    <h3 class="project-card__title">Local Peer Review</h3>
    <p class="project-card__desc">Automated academic peer-review generator that verifies citations, runs adversarial probes, and produces structured referee reports. Supports Ollama, OpenAI, Groq, and LM Studio backends with a Gradio web interface.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">LLM</span>
      <span class="project-card__tag">Ollama</span>
      <span class="project-card__tag">Gradio</span>
      <span class="project-card__tag">Python</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/local-peer-review" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-file-signature"></i></div>
      <div class="project-card__category">LLM Application</div>
    </div>
    <h3 class="project-card__title">Resume Customizer</h3>
    <p class="project-card__desc">Automatically tailors LaTeX resumes to match job descriptions using the Google Gemini API, with structured output parsing and PDF generation.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Gemini API</span>
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">LaTeX</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/resume_customizer" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-project-diagram"></i></div>
      <div class="project-card__category">LLM Tooling</div>
    </div>
    <h3 class="project-card__title">LaTeX for LLM</h3>
    <p class="project-card__desc">Converts LaTeX documents into a typed graph for targeted LLM context retrieval, where graph-based queries cut prompt size by ~54% on average (up to ~80% for focused queries). Features cross-reference resolution, inline math preservation, interactive force-directed graph visualization, and a Claude MCP server.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">MCP</span>
      <span class="project-card__tag">Graph</span>
      <span class="project-card__tag">NLP</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/LatexForLLM" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-rss"></i></div>
      <div class="project-card__category">Web App</div>
    </div>
    <h3 class="project-card__title">arXiv RSS Filter</h3>
    <p class="project-card__desc">Streamlit app that filters and curates arXiv preprint RSS feeds, helping researchers stay on top of relevant new papers in their field.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Streamlit</span>
      <span class="project-card__tag">RSS</span>
      <span class="project-card__tag">Python</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/RSS_arXiv" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

</div>

<h2 class="section-heading" style="margin-top: 3rem;"><span class="section-heading__accent">//</span> Machine Learning</h2>

<div class="projects-grid">

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-bicycle"></i></div>
      <div class="project-card__category">MLOps</div>
    </div>
    <h3 class="project-card__title">Seoul Bike Demand MLOps Pipeline</h3>
    <p class="project-card__desc">End-to-end, reproducible ML system for hourly bike-demand forecasting, built to run entirely on a laptop. A DVC pipeline chains ingestion, Pandera validation, feature engineering, LightGBM training, and evaluation; MLflow tracks every run and a gated registry only promotes a new champion when holdout RMSE improves. The champion is exported and served through a Dockerized FastAPI app, with Evidently drift monitoring wired to a retrain signal and CI on GitHub Actions. Holdout R&sup2; &asymp; 0.80 on a time-ordered split.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">MLflow</span>
      <span class="project-card__tag">DVC</span>
      <span class="project-card__tag">Docker</span>
      <span class="project-card__tag">Evidently</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/seoul-bike-mlops" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-brain"></i></div>
      <div class="project-card__category">Deep Learning</div>
    </div>
    <h3 class="project-card__title">Galaxy Morphology Deep Learning</h3>
    <p class="project-card__desc">PyTorch image classification of galaxy morphology on Galaxy10 SDSS: 21,785 real SDSS cutouts across 10 heavily imbalanced classes. Benchmarks a from-scratch CNN against ResNet-18 transfer learning, both frozen-feature extraction and full fine-tuning. Class-weighted loss and macro-F1 model selection handle the imbalance honestly; the fine-tuned ResNet-18 reaches &asymp; 0.80 test accuracy and 0.64 macro-F1. MLflow tracks every run and a Dockerized FastAPI service returns per-class predictions, with CI on GitHub Actions.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">PyTorch</span>
      <span class="project-card__tag">CNN</span>
      <span class="project-card__tag">Transfer Learning</span>
      <span class="project-card__tag">MLflow</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/galaxy_morphology_cnn" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-car-crash"></i></div>
      <div class="project-card__category">Classification</div>
    </div>
    <h3 class="project-card__title">UK Accident Severity Classification</h3>
    <p class="project-card__desc">Dual-strategy LightGBM classifiers on 104K DfT STATS19 collisions merged with 190K vehicle records (42 features). Severe-optimized model (0.527 macro recall, 66.8% accuracy) with threshold tuning for adjustable recall-accuracy tradeoff. Results benchmarked against published studies on the same dataset.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">LightGBM</span>
      <span class="project-card__tag">Feature Engineering</span>
      <span class="project-card__tag">Imbalanced Classification</span>
      <span class="project-card__tag">Streamlit</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/UK_road_safety_modelling" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
      <a href="https://uk-road-safety-modelling.streamlit.app/" class="project-card__link"><i class="fas fa-external-link-alt"></i> Live App</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-traffic-light"></i></div>
      <div class="project-card__category">Reinforcement Learning</div>
    </div>
    <h3 class="project-card__title">RL Traffic Signal Control</h3>
    <p class="project-card__desc">Deep RL agent that learns adaptive signal timing policies for a single intersection using real traffic flow data from the Transport for London API, replacing fixed-cycle control with a learned policy trained via PyTorch.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Deep RL</span>
      <span class="project-card__tag">PyTorch</span>
      <span class="project-card__tag">TfL API</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/RL_signal_control" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-brain"></i></div>
      <div class="project-card__category">Deep Learning</div>
    </div>
    <h3 class="project-card__title">Deep Reinforcement Learning</h3>
    <p class="project-card__desc">Implementations from Deep RL in Action covering policy gradients, Q-learning, and multi-agent environments using PyTorch.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">PyTorch</span>
      <span class="project-card__tag">Deep RL</span>
      <span class="project-card__tag">Neural Nets</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/Deep_RL" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

</div>

<h2 class="section-heading" style="margin-top: 3rem;"><span class="section-heading__accent">//</span> Quantitative Finance</h2>

<div class="projects-grid">

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-shield-alt"></i></div>
      <div class="project-card__category">Risk Analytics</div>
    </div>
    <h3 class="project-card__title">Risk Management Toolkit</h3>
    <p class="project-card__desc">End-to-end risk analytics pipeline: raw price data is denoised with Kalman filters and wavelet transforms, fed into technical indicators (MACD, Bollinger, PSAR, stochastic oscillators), then evaluated via historical &amp; parametric VaR, CAPM beta, and Sharpe/Treynor/Jensen performance metrics.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">VaR</span>
      <span class="project-card__tag">CAPM</span>
      <span class="project-card__tag">Kalman Filter</span>
      <span class="project-card__tag">Wavelets</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/risk-management" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-chart-line"></i></div>
      <div class="project-card__category">Time Series</div>
    </div>
    <h3 class="project-card__title">Time Series Forecasting</h3>
    <p class="project-card__desc">Comparative equity modeling: ARIMA for trend forecasting, GARCH for volatility clustering, CCC-GARCH for multi-asset portfolio-level forecasting, and Prophet for seasonality. Includes rolling &amp; fixed window evaluation, bootstrap simulation, and model diagnostics on Yahoo Finance data.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">ARIMA</span>
      <span class="project-card__tag">GARCH</span>
      <span class="project-card__tag">Prophet</span>
      <span class="project-card__tag">Forecasting</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/Time_Series_Analysis" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-sync-alt"></i></div>
      <div class="project-card__category">Backtesting</div>
    </div>
    <h3 class="project-card__title">Backtesting Framework</h3>
    <p class="project-card__desc">Strategy backtesting engine built on VectorBT's vectorized execution, progressing from moving-average crossover strategies to volatility-regime-aware ML signals incorporating GARCH-estimated conditional variance.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">VectorBT</span>
      <span class="project-card__tag">GARCH</span>
      <span class="project-card__tag">ML Signals</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/Backtesting" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-exchange-alt"></i></div>
      <div class="project-card__category">Algorithmic Trading</div>
    </div>
    <h3 class="project-card__title">ML Trading Signals</h3>
    <p class="project-card__desc">Momentum-based trading signal generation using MA ribbons, distance metrics, correlation-based signals, and physics-inspired momentum measurements.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">ML</span>
      <span class="project-card__tag">Momentum</span>
      <span class="project-card__tag">Signal Processing</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/ML_trading" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-chart-pie"></i></div>
      <div class="project-card__category">Portfolio Theory</div>
    </div>
    <h3 class="project-card__title">AI for Trading</h3>
    <p class="project-card__desc">Udacity nanodegree projects: momentum strategy, breakout strategy, and smart beta portfolio optimization with performance evaluation.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Momentum</span>
      <span class="project-card__tag">Smart Beta</span>
      <span class="project-card__tag">Portfolio</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/AI_for_trading" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-coins"></i></div>
      <div class="project-card__category">Web App</div>
    </div>
    <h3 class="project-card__title">Mutual Fund Analyzer</h3>
    <p class="project-card__desc">Live Streamlit Cloud app that ingests Morningstar fund data to build multi-fund portfolios and analyze them across sectoral distribution, scheme comparison, valuations, and top company exposure.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Streamlit</span>
      <span class="project-card__tag">Altair</span>
      <span class="project-card__tag">Scikit-learn</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/MF_analysis" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
      <a href="https://mf-analysis.streamlit.app/" class="project-card__link"><i class="fas fa-external-link-alt"></i> Live App</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-chart-bar"></i></div>
      <div class="project-card__category">Web App</div>
    </div>
    <h3 class="project-card__title">NSE Stocks Dashboard</h3>
    <p class="project-card__desc">Streamlit dashboard fetching live data from the NSE API for real-time stock analysis, price monitoring, and technical charting of equities listed on the National Stock Exchange of India.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Streamlit</span>
      <span class="project-card__tag">Finance</span>
      <span class="project-card__tag">Dashboard</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/Streamlit_stocks_analysis" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

</div>

<h2 class="section-heading" style="margin-top: 3rem;"><span class="section-heading__accent">//</span> Astrophysics Research</h2>

<div class="projects-grid">

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-satellite"></i></div>
      <div class="project-card__category">Astrophysics</div>
    </div>
    <h3 class="project-card__title">AGN &amp; Local Environment in HR5</h3>
    <p class="project-card__desc">Investigated effects of local environment on Active Galactic Nuclei activity using the Horizon Run 5 cosmological simulation with Python-based feature extraction pipelines on terabyte-scale data.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">HPC</span>
      <span class="project-card__tag">Simulations</span>
    </div>
    <div class="project-card__links">
      <a href="https://ui.adsabs.harvard.edu/abs/2023ApJ...953...64S" class="project-card__link"><i class="fas fa-file-alt"></i> Paper</a>
      <a href="https://github.com/ansingh16/hr5_metallicity" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-sun"></i></div>
      <div class="project-card__category">Astrophysics</div>
    </div>
    <h3 class="project-card__title">Dust Radiative Transfer (HR5)</h3>
    <p class="project-card__desc">Modified the Powderday radiative transfer code for Horizon Run 5 AMR data. Built pipelines for galaxy selection, sub-cube extraction from HDF5 snapshots, and SED computation with MPI parallelization.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">Cython</span>
      <span class="project-card__tag">MPI</span>
      <span class="project-card__tag">HDF5</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/ansingh16/hr5_agn_sed" class="project-card__link"><i class="fab fa-github"></i> GitHub</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-globe"></i></div>
      <div class="project-card__category">Astrophysics</div>
    </div>
    <h3 class="project-card__title">Cosmic Web Analysis at Gigaparsec Scales</h3>
    <p class="project-card__desc">Developed a multi-block approach for the DisPerSE cosmic web finder enabling analysis at unprecedented gigaparsec scales.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">C</span>
      <span class="project-card__tag">HPC</span>
      <span class="project-card__tag">Algorithms</span>
    </div>
    <div class="project-card__links">
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__header">
      <div class="project-card__icon"><i class="fas fa-wind"></i></div>
      <div class="project-card__category">Astrophysics</div>
    </div>
    <h3 class="project-card__title">Ram Pressure Stripping Models</h3>
    <p class="project-card__desc">Analytical and simulation-based study of ram pressure stripping dependence on orbital parameters of galaxies.</p>
    <div class="project-card__tags">
      <span class="project-card__tag">Python</span>
      <span class="project-card__tag">Analytics</span>
      <span class="project-card__tag">Simulations</span>
    </div>
    <div class="project-card__links">
      <a href="https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.5582S" class="project-card__link"><i class="fas fa-file-alt"></i> Paper</a>
    </div>
  </div>

</div>
