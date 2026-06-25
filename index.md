---
layout: page
permalink: /
title: Babs Olaniyi
---

<section class="hero">
  <div class="hero__content">
    <p class="eyebrow">Senior Data Scientist / AI-ML Engineer</p>
    <h1>Machine learning systems for healthcare AI, pricing, recommendations, and experimentation.</h1>
    <p class="lead">
      I am Babs Olaniyi, a data scientist at Causal Foundry in Barcelona. I build models, experiments, and data products that support healthcare financing, provider performance, pricing decisions, user engagement, and operational decision-making.
    </p>
    <div class="hero__actions">
      <a class="button button--primary" href="/portfolio/">View projects</a>
      <a class="button" href="/cv/">Read CV</a>
      <a class="button" href="https://www.linkedin.com/in/babaniyi/">LinkedIn</a>
    </div>
  </div>
  <div class="hero__panel">
    <img src="/images/profile.jpeg" alt="Babaniyi Olaniyi" class="hero__portrait">
    <div>
      <p class="panel-label">Current focus</p>
      <p>Current work: healthcare AI, claims analytics, provider performance, capitation reform, anomaly detection, and decision-support systems at Causal Foundry.</p>
    </div>
  </div>
</section>

<section class="section-grid">
  <div class="metric">
    <span class="metric__value">7+</span>
    <span class="metric__label">years across data science, ML, and analytics</span>
  </div>
  <div class="metric">
    <span class="metric__value">9M+</span>
    <span class="metric__label">insured individuals represented in healthcare analytics work</span>
  </div>
  <div class="metric">
    <span class="metric__value">EUR11M+</span>
    <span class="metric__label">annual pricing leakage protected through anomaly detection</span>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Selected Work</p>
    <h2>Recent projects and research</h2>
  </div>
  <div class="card-grid">
    <article class="feature-card">
      <p class="card-meta">Healthcare AI</p>
      <h3>Claims-driven health financing</h3>
      <p>Built models and pipelines for healthcare financing, provider performance, capitation reform, anomaly detection, and decision support across large-scale claims data.</p>
    </article>
    <article class="feature-card">
      <p class="card-meta">Pricing and revenue</p>
      <h3>Pricing leakage and optimization</h3>
      <p>Developed anomaly detection, elasticity, uplift, and optimal-pricing models across a EUR500M+ product portfolio.</p>
    </article>
    <article class="feature-card">
      <p class="card-meta">AI/ML engineering</p>
      <h3>Multimodal LLM recommendations</h3>
      <p>Built recommendation experiments that combine product reviews, metadata, images, temporal signals, and ranking metrics such as Recall, MRR, Hit Rate, and NDCG.</p>
      <a href="https://github.com/babaniyi/MultiModal-LLM-RecSys">Open project</a>
    </article>
  </div>
</section>

<section class="content-section split-section">
  <div>
    <p class="eyebrow">Writing</p>
    <h2>Latest posts</h2>
    <ul class="clean-list">
      {% for post in site.posts limit:5 %}
        <li>
          <a href="{{ post.url }}">{{ post.title }}</a>
          <span>{{ post.date | date: "%b %Y" }}</span>
        </li>
      {% endfor %}
    </ul>
  </div>
  <div>
    <p class="eyebrow">Publications</p>
    <h2>Research</h2>
    <ul class="clean-list">
      <li><a href="https://arxiv.org/abs/2510.21851">Data-Driven Approach to Capitation Reform in Rwanda</a><span>2025</span></li>
      <li><span>Enhancing Product Recommendations with Multi-Modal LLMs</span><span>ISIR-eCom 2025</span></li>
      <li><a href="https://arxiv.org/abs/2206.08178">User Engagement in Mobile Health Applications</a><span>KDD 2022</span></li>
    </ul>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Skills</p>
    <h2>Working toolkit</h2>
  </div>
  <div class="tag-cloud">
    <span>Python</span>
    <span>SQL</span>
    <span>PySpark</span>
    <span>Scala</span>
    <span>Google Cloud</span>
    <span>BigQuery</span>
    <span>Azure Databricks</span>
    <span>MLflow</span>
    <span>Airflow</span>
    <span>AWS</span>
    <span>PyTorch</span>
    <span>TensorFlow</span>
    <span>XGBoost</span>
    <span>LightGBM</span>
    <span>Scikit-learn</span>
    <span>A/B testing</span>
    <span>Forecasting</span>
    <span>Survival analysis</span>
    <span>Causal inference</span>
    <span>Recommender systems</span>
  </div>
</section>
