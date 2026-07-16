---
permalink: /
title: "Ankit Singh"
layout: home
author_profile: false
hero_bio: "Data Scientist with a PhD in Physics. I build ML models, data pipelines, and deployed apps, with a background in statistical modeling and terabyte-scale scientific computing."
skills_programming:
  - Python
  - SQL
  - C
  - Cython
  - Bash
  - Streamlit
  - Gradio
  - Flask
  - Git
  - Linux
  - HPC (MPI/Slurm)
  - Jupyter
  - Streamlit Cloud
  - GitHub Pages
skills_ml:
  - PyTorch
  - Scikit-learn
  - LightGBM
  - CNNs
  - LSTMs
  - Transfer Learning
  - GANs
  - YOLOv8
  - ARIMA
  - GARCH
  - Prophet
  - SMOTE/ADASYN
  - K-Means
  - PCA
  - OpenCV
skills_genai:
  - RAG Pipelines
  - FAISS
  - ChromaDB
  - Sentence-Transformers
  - OpenAI API
  - Gemini API
  - Ollama
  - LLM Engineering
skills_astro:
  - Galaxy Evolution
  - Cosmological Simulations
  - SED Fitting
  - Monte Carlo Methods
  - Radiative Transfer
  - Large-scale Structure
  - AGN
featured_projects:
  - title: "UK Accident Severity Classification"
    description: "Dual-strategy ML system on 151K UK road accidents: emergency response model achieving 92.4% severe recall and traffic management model with 81% macro recall using SMOTE+Tomek and ADASYN."
    tags: ["LightGBM", "Scikit-learn", "SMOTE"]
    icon: "fas fa-car-crash"
    github: "https://github.com/ansingh16/UK_road_safety_modelling"
  - title: "UK Visa RAG Chatbot"
    description: "RAG chatbot using FAISS/ChromaDB vector stores and sentence-transformers to answer UK immigration questions from GOV.UK data."
    tags: ["RAG", "FAISS", "Streamlit", "NLP"]
    icon: "fas fa-robot"
    github: "https://github.com/ansingh16/UK_visa_RAG"
  - title: "M25 Congestion Forecasting with LSTM"
    description: "PyTorch LSTM forecasting motorway congestion from real Transport for London sensor data, reading a 4-hour window of 15-minute volume and speed readings to predict the next interval. Congestion is the minority class, rising from 5.5% of training intervals to 25.6% in the chronologically held-out period. Benchmarked against majority-class, rush-hour and speed-threshold baselines: 0.977 recall, 0.765 F1, PR-AUC 0.893 against a no-skill 0.256."
    tags: ["PyTorch", "LSTM", "Time Series", "API"]
    icon: "fas fa-traffic-light"
    github: "https://github.com/ansingh16/tfl-congestion-lstm"
  - title: "Mutual Fund Analyzer"
    description: "Deployed Streamlit app with Morningstar data for portfolio analysis covering sectoral distribution, Sharpe ratio, and company exposure."
    tags: ["Streamlit", "Altair", "Scikit-learn"]
    icon: "fas fa-coins"
    github: "https://github.com/ansingh16/MF_analysis"
    demo: "https://mf-analysis.streamlit.app/"
redirect_from:
  - /about/
  - /about.html
---

I'm a Research Fellow at the **University of Nottingham** with a PhD in Physics and 10+ years working with terabyte-scale simulation data, statistical modeling, and HPC. Recent projects include an end-to-end MLOps pipeline with drift monitoring, deep learning for galaxy morphology, a dual-model classifier for 104K UK road accidents, an LSTM congestion forecaster on TfL sensor data, RAG chatbots, and quantitative finance tools.

I'm an ISO/IEC 42001 certified practitioner (AI Management Systems) and AI+ Foundation certified, with additional applied data science credentials from WorldQuant University. 12 peer-reviewed publications, 100+ citations.
