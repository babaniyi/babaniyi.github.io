---
layout: page
permalink: /
title: Babaniyi Olaniyi
---

<section class="hero">
  <div class="hero__content">
    <p class="eyebrow">Senior Data Scientist</p>
    <h1>Data science for pricing, recommendation systems, experimentation, and health technology.</h1>
    <p class="lead">
      I am Babaniyi Olaniyi, a Senior Data Scientist currently working as a Data Scientist. I build machine learning systems that turn messy product, commercial, and health data into decisions people can use.
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
      <p>Machine learning systems for pricing, demand forecasting, recommender systems, product experimentation, and digital health.</p>
    </div>
  </div>
</section>

<section class="section-grid">
  <div class="metric">
    <span class="metric__value">5+</span>
    <span class="metric__label">years in applied data science</span>
  </div>
  <div class="metric">
    <span class="metric__value">EUR40M+</span>
    <span class="metric__label">revenue impact from pricing and leakage work</span>
  </div>
  <div class="metric">
    <span class="metric__value">50K+</span>
    <span class="metric__label">medications forecasted for health supply chains</span>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Selected Work</p>
    <h2>Recent projects and research</h2>
  </div>
  <div class="card-grid">
    <article class="feature-card">
      <p class="card-meta">Pricing and revenue</p>
      <h3>Customer pricing leakage detection</h3>
      <p>Built heuristic and machine learning approaches to identify price arbitrage, correct pricing errors, and improve gross profit against baseline price scenarios.</p>
    </article>
    <article class="feature-card">
      <p class="card-meta">Health technology</p>
      <h3>Demand forecasting for pharmacies</h3>
      <p>Forecasted demand across more than 50,000 medications to improve stock planning and reduce patient wait times in pharmacy networks.</p>
    </article>
    <article class="feature-card">
      <p class="card-meta">Recommender systems</p>
      <h3>LLMs for product recommendations</h3>
      <p>Explored how large language models can generate personalized recommendations from product metadata and reviews.</p>
      <a href="https://github.com/babaniyi/LLMs-for-RecSys">Open project</a>
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
      <li><a href="https://arxiv.org/abs/2206.08178">User Engagement in Mobile Health Applications</a><span>KDD 2022</span></li>
      <li><span>Power Samade distribution: properties and application to real lifetime data</span><span>2023</span></li>
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
    <span>PyTorch</span>
    <span>TensorFlow</span>
    <span>Scikit-learn</span>
    <span>A/B testing</span>
    <span>Forecasting</span>
    <span>Survival analysis</span>
    <span>Causal inference</span>
    <span>Recommender systems</span>
  </div>
</section>
