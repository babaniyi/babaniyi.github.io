---
layout: page
title: Portfolio
permalink: /portfolio/
---

<section class="page-intro">
  <p class="eyebrow">Portfolio</p>
  <h1>Senior data science and AI/ML engineering work across healthcare AI, pricing, recommender systems, and experimentation.</h1>
  <p class="lead">
    A focused view of work that best represents my profile: production-minded machine learning, applied research, data products, and business-facing analytics with measurable impact.
  </p>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Featured case studies</p>
    <h2>High-impact applied work</h2>
  </div>
  <div class="case-study-grid">
    <article class="case-study">
      <p class="card-meta">Healthcare AI / Decision Support</p>
      <h3>Causal Foundry: claims analytics and health financing models</h3>
      <p>Built healthcare analytics and decision-support systems using claims and provider data to support financing reform, provider performance, anomaly detection, and operational monitoring.</p>
      <ul class="evidence-list">
        <li>Worked with datasets covering more than 9M insured individuals and 4,500+ healthcare facilities.</li>
        <li>Built large-scale pipelines processing 100M+ healthcare claims records using Python and SQL.</li>
        <li>Developed monitoring systems with anomaly detection and automated indicators across 1,000+ providers.</li>
        <li>Reconciled 5,000+ drug SKUs across seven agencies using text similarity and embedding-based matching.</li>
      </ul>
      <div class="tag-cloud tag-cloud--small">
        <span>Healthcare AI</span>
        <span>Claims analytics</span>
        <span>Provider performance</span>
        <span>Anomaly detection</span>
      </div>
    </article>

    <article class="case-study">
      <p class="card-meta">Pricing / Commercial ML</p>
      <h3>ZF Group: pricing leakage detection and optimization</h3>
      <p>Developed anomaly detection, elasticity, uplift, and optimization models for commercial pricing decisions across a large industrial product portfolio.</p>
      <ul class="evidence-list">
        <li>Protected approximately EUR11M annually by identifying pricing inconsistencies and channel leakage.</li>
        <li>Built pricing and uplift models across a EUR500M+ portfolio, contributing to a 13% gross-profit improvement.</li>
        <li>Deployed PySpark and Databricks pipelines with engineering partners for commercial analytics workflows.</li>
      </ul>
      <div class="tag-cloud tag-cloud--small">
        <span>Pricing</span>
        <span>XGBoost</span>
        <span>PySpark</span>
        <span>Databricks</span>
      </div>
    </article>

    <article class="case-study">
      <p class="card-meta">Digital Health / Experimentation</p>
      <h3>Mobile health engagement and demand forecasting</h3>
      <p>Built forecasting, survival, and experimentation workflows for pharmacy supply chains, healthcare workers, and mHealth applications.</p>
      <ul class="evidence-list">
        <li>Forecasted demand for pharmacy networks using DeepAR and time-series modeling, reducing stockouts by 18%.</li>
        <li>Improved engagement outcomes with survival modeling, churn prediction, contextual bandits, and rule-based optimization.</li>
        <li>Designed A/B testing, multi-armed bandit, and reinforcement-learning approaches for adaptive interventions.</li>
      </ul>
      <div class="tag-cloud tag-cloud--small">
        <span>Forecasting</span>
        <span>Survival analysis</span>
        <span>A/B testing</span>
        <span>Bandits</span>
      </div>
    </article>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">AI and recommender systems</p>
    <h2>Engineering projects</h2>
  </div>
  <section class="project-list">
    <article class="project-card project-card--wide">
      <img src="/images/projects/design.png" alt="Recommendation systems diagram">
      <div>
        <p class="card-meta">Multimodal LLMs / Recommenders</p>
        <h2><a href="https://github.com/babaniyi/MultiModal-LLM-RecSys">Multi-Modal LLM-based Product Recommender System</a></h2>
        <p>Built a recommendation system that combines product reviews, metadata, images, temporal ordering, and multimodal feature fusion to predict the next items a user may purchase or review.</p>
        <ul class="evidence-list">
          <li>Built a multimodal recommender over approximately 3M Amazon interactions using review text and image embeddings.</li>
          <li>Fine-tuned a GPT-2 style recommendation model with textual, visual, and temporal features.</li>
          <li>Implemented train/validation/test processing that respects time ordering to reduce leakage.</li>
          <li>Reported NDCG@5 of 0.22 and P@5 of 0.29, with evaluation harnesses, ablations, and documentation.</li>
        </ul>
        <div class="tag-cloud tag-cloud--small">
          <span>LLMs</span>
          <span>Computer vision</span>
          <span>Ranking metrics</span>
          <span>PyTorch</span>
        </div>
      </div>
    </article>

    <article class="project-card project-card--wide">
      <img src="/images/projects/markus-winkler-unsplash.jpg" alt="Decision optimization visual">
      <div>
        <p class="card-meta">Experimentation / Reinforcement learning</p>
        <h2><a href="https://github.com/babaniyi/Deep-contextual-bandits">Deep Contextual Bandits</a></h2>
        <p>Adapted deep contextual bandit ideas for reusable experimentation workflows, focusing on Bayesian neural networks, Thompson sampling, and decision-making under uncertainty.</p>
        <ul class="evidence-list">
          <li>Grounded in the ICLR 2018 Deep Bayesian Bandits benchmark.</li>
          <li>Built toward a package that can run contextual bandit algorithms on arbitrary datasets.</li>
          <li>Connects directly to product experimentation and adaptive intervention design.</li>
        </ul>
        <div class="tag-cloud tag-cloud--small">
          <span>Contextual bandits</span>
          <span>Thompson sampling</span>
          <span>TensorFlow</span>
          <span>Experimentation</span>
        </div>
      </div>
    </article>

    <article class="project-card project-card--wide">
      <img src="/images/projects/sales.jpeg" alt="Retail analytics dashboard">
      <div>
        <p class="card-meta">Customer analytics / ML</p>
        <h2><a href="https://github.com/babaniyi/BusinessML">Business ML: retention, attribution, and segmentation</a></h2>
        <p>Converted online retail transactions into a business analytics workflow covering revenue, retention, customer growth, attribution, journey analysis, and RFM-based segmentation.</p>
        <div class="tag-cloud tag-cloud--small">
          <span>Cohort analysis</span>
          <span>Attribution</span>
          <span>RFM</span>
          <span>Segmentation</span>
        </div>
      </div>
    </article>
  </section>
</section>

<section class="content-section split-section">
  <div>
    <p class="eyebrow">Publications</p>
    <h2>Research and papers</h2>
    <div class="article-list">
      <article>
        <span>ISIR-eCom 2025</span>
        <h3>Enhancing Product Recommendations with Multi-Modal LLMs</h3>
        <p>Research on multimodal product recommendation using text and image representations for next-item prediction.</p>
      </article>
      <article>
        <span>2025</span>
        <h3><a href="https://arxiv.org/abs/2510.21851">Data-Driven Approach to Capitation Reform in Rwanda</a></h3>
        <p>Claims-data-driven capitation design, calibration, monitoring, and prescribing-quality insights for Rwanda's Community-Based Health Insurance scheme.</p>
      </article>
      <article>
        <span>KDD 2022</span>
        <h3><a href="https://arxiv.org/abs/2206.08178">User Engagement in Mobile Health Applications</a></h3>
        <p>Probabilistic and survival-analysis framework for engagement and churn in mobile health applications used by healthcare workers.</p>
      </article>
      <article>
        <span>2023</span>
        <h3>Power Samade distribution: properties and application to real lifetime data</h3>
        <p>Nigerian Journal of Science and Environment paper on distributional modeling and lifetime data analysis.</p>
      </article>
      <article>
        <span>2018</span>
        <h3>Homework vs. In Class-Exercise: Means of Assessment, Waste of Time or Punishment?</h3>
        <p>International Journal of Scientific and Engineering Research.</p>
      </article>
    </div>
  </div>
  <div>
    <p class="eyebrow">Writing</p>
    <h2>Selected articles</h2>
    <div class="article-list">
      <article>
        <span>2023</span>
        <h3><a href="/2023/03/22/designing-a-recommendation-system-for-search-in-ecommerce.html">Designing Recommendation Systems for Search in E-commerce</a></h3>
        <p>System-design oriented discussion of retrieval, ranking, and search recommendation tradeoffs.</p>
      </article>
      <article>
        <span>2022</span>
        <h3><a href="/2022/07/16/designing-machine-learning-solution-for-course-recommendation.html">Designing Machine Learning Solution for Course Recommendation</a></h3>
        <p>End-to-end framing of a course recommendation problem from business goal to ML design.</p>
      </article>
      <article>
        <span>2020</span>
        <h3><a href="https://babaniyi.medium.com/customer-spend-satisfaction-and-segmentation-using-machine-learning-techniques-15822b60f5b">Customer spend, satisfaction, and segmentation</a></h3>
        <p>Marketplace analytics using customer segmentation, satisfaction prediction, and spend modeling.</p>
      </article>
      <article>
        <span>2019</span>
        <h3>Identifying networks in customer reviews</h3>
        <p>Network analysis applied to customer review relationships and behavioral insight discovery.</p>
      </article>
    </div>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Selected earlier work</p>
    <h2>Analytics breadth</h2>
  </div>
  <div class="card-grid card-grid--two">
    <article class="feature-card">
      <h3><a href="https://nbviewer.jupyter.org/github/neahyo/Metyis/blob/2ab5b24e901cf1eaa1dbcb657684ebc311ff0882/Metyis/Analysis.ipynb">Is the movie industry dying?</a></h3>
      <p>Explored film revenue, budget, genre, rating, audience, cast, and director effects to advise production strategy.</p>
    </article>
    <article class="feature-card">
      <h3>The determinants of happiness</h3>
      <p>Applied statistical modeling to study drivers of subjective wellbeing and socioeconomic outcomes.</p>
    </article>
    <article class="feature-card">
      <h3>Is comparison really the thief of joy?</h3>
      <p>Empirical analysis using South African data to study comparison, life satisfaction, and economic context.</p>
    </article>
    <article class="feature-card">
      <h3>Data visualization practice</h3>
      <p>Built a reference collection of Python visualization patterns inspired by The Economist, data-to-viz, and storytelling-with-data practices.</p>
    </article>
  </div>
</section>
