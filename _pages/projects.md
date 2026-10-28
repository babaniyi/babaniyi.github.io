---
layout: page
title: Work
permalink: /portfolio/
description: Selected projects, publications, and applied data science work by Babs Olaniyi, across healthcare, public policy, pricing, and machine learning.
---

<section class="page-hero">
  <p class="eyebrow">Portfolio</p>
  <h1>Selected work</h1>
  <div class="page-hero__meta"><p class="lead">Projects from healthcare, public policy, pricing, and recommendation systems.</p><p>A mix of industry work, research, and open-source projects.</p></div>
</section>

<section class="content-section" aria-labelledby="featured-project">
  <div class="section-kicker"><p class="eyebrow">Case study</p><span>01</span></div>
  <div class="project-feature">
    <div>
      <p class="card-meta">Development economics / Applied NLP</p>
      <h2 id="featured-project"><a href="{{ '/portfolio/food-crisis-early-warning/' | relative_url }}">Food-crisis early warning from bilingual news</a></h2>
      <p>I designed a research prototype that turns English and Arabic news into location-level evidence for food-security analysts across Iraq, Jordan, Lebanon, Palestine, and Syria.</p>
      <p>The system matches 167 risk factors across 12 themes and resolves mentions against 357 geographic entities. Analysts can trace every flag back to its article, source, language, and location.</p>
      <p>Signal strength and evidence confidence are reported separately. The output is a queue for investigation, with a plan for future outcome validation.</p>
      <a class="text-link" href="{{ '/portfolio/food-crisis-early-warning/' | relative_url }}">Read the case study <span aria-hidden="true">&rarr;</span></a>
    </div>
    <figure>
      <a href="{{ '/portfolio/food-crisis-early-warning/' | relative_url }}"><img src="{{ '/images/projects/food-crisis-early-warning/location-verification-queue.png' | relative_url }}" alt="Illustrative analyst verification queue showing separate signal and confidence measures" width="1107" height="538" loading="lazy"></a>
      <figcaption>Illustrative verification output. The prototype prioritizes analyst review; these scores do not establish crisis probabilities.</figcaption>
    </figure>
  </div>
</section>

<section class="content-section" aria-label="Industry work">
  <div class="section-kicker"><p class="eyebrow">Industry work</p><span>02</span></div>
  <div class="applied-work">
    <article id="health-financing">
      <div><p class="card-meta">Causal Foundry / Healthcare</p><h2>Claims-driven health financing and provider performance</h2></div>
      <div><p>I build analytical models and monitoring systems for financing reform, utilization, prescribing quality, and provider performance.</p><ul class="evidence-list"><li>Claims pipelines processing more than 100M records, with data representing over 9M insured people.</li><li>Automated indicators and anomaly detection for oversight across 1,000+ providers.</li><li>Reconciliation of more than 5,000 drug SKUs across seven agencies using text similarity and embeddings.</li></ul><a class="text-link" href="https://arxiv.org/abs/2510.21851">Read the capitation reform paper <span aria-hidden="true">&nearr;</span></a></div>
    </article>
    <article id="pricing">
      <div><p class="card-meta">ZF Group / Commercial ML</p><h2>Pricing leakage detection and optimization</h2></div>
      <div><p>I developed anomaly-detection, elasticity, uplift, and optimization models to support commercial pricing decisions.</p><ul class="evidence-list"><li>Pricing-arbitrage detection that protected approximately &euro;11M in annual leakage.</li><li>Models across a &euro;500M+ product portfolio, contributing to a 13% gross-profit improvement.</li><li>PySpark and Databricks pipelines developed with engineering partners.</li></ul></div>
    </article>
    <article id="digital-health">
      <div><p class="card-meta">Benshi AI / Digital health</p><h2>Engagement, forecasting, and adaptive interventions</h2></div>
      <div><p>I worked on forecasting and experimentation for pharmacy supply chains, healthcare workers, and mobile-health applications.</p><ul class="evidence-list"><li>Demand forecasts that helped reduce pharmacy stockouts by 18%.</li><li>Survival models and churn prediction to understand engagement.</li><li>Adaptive interventions using contextual bandits and reinforcement-learning concepts.</li></ul><a class="text-link" href="https://arxiv.org/abs/2206.08178">Read the KDD engagement paper <span aria-hidden="true">&nearr;</span></a></div>
    </article>
  </div>
</section>

<section class="content-section" aria-label="Engineering and research projects">
  <div class="section-kicker"><p class="eyebrow">Research &amp; code</p><span>03</span></div>
  <div class="project-notes">
    <article><p class="card-meta">Recommender systems / Multimodal learning</p><h3><a href="https://github.com/babaniyi/MultiModal-LLM-RecSys">Multi-Modal LLM Product Recommender</a></h3><p>Review text, metadata, images, and temporal signals for next-item prediction across approximately 3M interactions. The project includes evaluation and ablation workflows.</p><a class="text-link" href="https://ieeexplore.ieee.org/abstract/document/11416058/">Related publication <span aria-hidden="true">&nearr;</span></a></article>
    <article><p class="card-meta">Experimentation / Decision-making</p><h3><a href="https://github.com/babaniyi/Deep-contextual-bandits">Deep Contextual Bandits</a></h3><p>Experimentation workflows built around Bayesian neural networks, Thompson sampling, and decisions under uncertainty, adapting ideas from the Deep Bayesian Bandits benchmark.</p><a class="text-link" href="https://github.com/babaniyi/Deep-contextual-bandits">Source code <span aria-hidden="true">&nearr;</span></a></article>
    <article><p class="card-meta">Customer analytics</p><h3><a href="https://github.com/babaniyi/BusinessML">Segmentation and retention intelligence</a></h3><p>Online retail data used to explore retention, growth, attribution, customer journeys, and RFM segmentation.</p><a class="text-link" href="https://github.com/babaniyi/BusinessML">Source code <span aria-hidden="true">&nearr;</span></a></article>
    <article><p class="card-meta">NLP / Research</p><h3>Text as evidence</h3><p>Work on retrieval, classification, similarity, and network analysis for public-policy and customer-behavior questions, including newspaper-based research on political violence at IAE-CSIC.</p><a class="text-link" href="{{ '/about/' | relative_url }}">Background and experience <span aria-hidden="true">&rarr;</span></a></article>
  </div>
</section>

<section class="closing-cta"><h2>Background and research</h2><div><a class="text-link" href="{{ '/about/' | relative_url }}">Experience &amp; education <span aria-hidden="true">&rarr;</span></a><a class="text-link" href="{{ '/#publications' | relative_url }}">Papers &amp; publications <span aria-hidden="true">&rarr;</span></a></div></section>
