---
layout: page
permalink: /
title: Babs Olaniyi
description: Babs Olaniyi, data scientist and economist in Barcelona. Work on healthcare financing, experimentation, pricing, and machine learning.
---

<section class="personal-intro" id="profile" aria-labelledby="intro-title">
  <div class="personal-intro__identity">
    <div>
      <p class="eyebrow">Data scientist &amp; economist / Barcelona</p>
      <h1 id="intro-title">Babs Olaniyi<span class="name-period">.</span></h1>
      <p class="intro-subtitle">Models, experiments, and the decisions behind them.</p>
    </div>
    <img src="{{ '/images/profile.jpeg' | relative_url }}" alt="Babs Olaniyi" class="personal-intro__portrait" width="168" height="190" fetchpriority="high">
  </div>
  <div class="personal-intro__body">
    <p>I am a data scientist at Causal Foundry. My work brings together economics, statistics, and machine learning, from healthcare financing in Rwanda to pricing and recommendation systems.</p>
    <p>I like working through the whole problem: what to measure, how to test it, and how to turn the result into something people can use.</p>
    <div class="inline-links"><a href="{{ '/about/' | relative_url }}">More about me <span aria-hidden="true">&rarr;</span></a><a href="https://github.com/babaniyi">GitHub <span aria-hidden="true">&nearr;</span></a><a href="https://www.linkedin.com/in/babaniyi/">LinkedIn <span aria-hidden="true">&nearr;</span></a></div>
  </div>
</section>

<section class="index-section" aria-labelledby="work-heading">
  <div class="index-section__heading"><h2 id="work-heading">Selected work</h2><a href="{{ '/portfolio/' | relative_url }}">Full portfolio <span aria-hidden="true">&rarr;</span></a></div>
  <div class="work-index">
    <article class="work-entry">
      <p class="work-entry__meta">Public policy<br><span>Research prototype</span></p>
      <div><h3><a href="{{ '/portfolio/food-crisis-early-warning/' | relative_url }}">Food-crisis early warning from bilingual news <span aria-hidden="true">&rarr;</span></a></h3><p>English and Arabic news turned into an auditable review queue for food-security analysts across five Mashriq countries.</p></div>
    </article>
    <article class="work-entry">
      <p class="work-entry__meta">Healthcare<br><span>Causal Foundry</span></p>
      <div><h3><a href="{{ '/portfolio/#health-financing' | relative_url }}">Claims-driven health financing <span aria-hidden="true">&rarr;</span></a></h3><p>Claims analytics, capitation reform, and provider monitoring using data representing more than 9 million insured people.</p></div>
    </article>
    <article class="work-entry">
      <p class="work-entry__meta">Pricing<br><span>ZF Group</span></p>
      <div><h3><a href="{{ '/portfolio/#pricing' | relative_url }}">Finding and preventing pricing leakage <span aria-hidden="true">&rarr;</span></a></h3><p>Anomaly detection, elasticity, and uplift models across a &euro;500M+ industrial product portfolio.</p></div>
    </article>
    <article class="work-entry">
      <p class="work-entry__meta">Recommendations<br><span>Research &amp; code</span></p>
      <div><h3><a href="https://github.com/babaniyi/MultiModal-LLM-RecSys">Product recommendations with multimodal models <span aria-hidden="true">&nearr;</span></a></h3><p>Combining reviews, images, and temporal signals for next-item prediction across approximately 3 million interactions.</p></div>
    </article>
  </div>
</section>

<section class="index-section" aria-labelledby="research-heading">
  <div class="index-section__heading"><h2 id="research-heading">Papers &amp; publications</h2><a href="{{ '/cv/#publications' | relative_url }}">All publications <span aria-hidden="true">&rarr;</span></a></div>
  <div class="publication-list">
    <article><span>2025</span><div><h3><a href="https://arxiv.org/abs/2510.21851">Data-Driven Approach to Capitation Reform in Rwanda</a></h3><p>Health financing, claims data, and provider payment.</p></div></article>
    <article><span>2025</span><div><h3><a href="https://ieeexplore.ieee.org/abstract/document/11416058/">Enhancing Product Recommendations with Multi-Modal LLMs</a></h3><p>ISIR-eCom / Text and image signals for recommendations.</p></div></article>
    <article><span>2022</span><div><h3><a href="https://arxiv.org/abs/2206.08178">User Engagement in Mobile Health Applications</a></h3><p>KDD / Probabilistic and survival models of engagement.</p></div></article>
  </div>
</section>

<section class="index-section" aria-labelledby="writing-heading">
  <div class="index-section__heading"><h2 id="writing-heading">Writing</h2><a href="{{ '/posts/' | relative_url }}">All writing <span aria-hidden="true">&rarr;</span></a></div>
  <ul class="essay-index">
    {% for post in site.posts limit:3 %}
    <li><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y" }}</time><a href="{{ post.url | relative_url }}">{{ post.title | escape }} <span aria-hidden="true">&rarr;</span></a></li>
    {% endfor %}
    {% for article in site.data.medium_articles %}
    {% unless article.local_url %}
    <li><time datetime="{{ article.date }}">{{ article.date | date: "%Y" }}</time><a href="{{ article.url }}">{{ article.title | escape }} <span aria-hidden="true">&nearr;</span></a></li>
    {% endunless %}
    {% endfor %}
  </ul>
  <p class="section-footnote">I also publish on <a href="https://medium.com/@babaniyi">Medium</a> and <a href="https://babaniyi.substack.com">Substack</a>.</p>
</section>

<section class="index-section" aria-labelledby="reading-heading">
  <div class="index-section__heading"><h2 id="reading-heading">Favourite reads</h2><span>A few bookmarks</span></div>
  <p class="reading-intro">Ideas I return to when thinking about experiments, recommendations, and building software.</p>
  <div class="reading-columns">
    <ul class="reading-list">
      <li><a href="https://terriblesoftware.org/2026/03/03/nobody-gets-promoted-for-simplicity/">Nobody Gets Promoted for Simplicity</a><span>Terrible Software / Engineering</span></li>
      <li><a href="https://eugeneyan.com/writing/bandits/">Bandits for Recommender Systems</a><span>Eugene Yan / Experimentation</span></li>
      <li><a href="https://magazine.sebastianraschka.com/p/understanding-multimodal-llms">Understanding Multimodal LLMs</a><span>Sebastian Raschka / Machine learning</span></li>
    </ul>
    <ul class="reading-list">
      <li><a href="https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html">Data Distribution Shifts and Monitoring</a><span>Chip Huyen / ML systems</span></li>
      <li><a href="https://merely-useful.tech/py-rse/index.html">Research Software Engineering in Python</a><span>Research practice</span></li>
      <li><a href="https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/">Multi-Armed Bandits and the Stitch Fix Experimentation Platform</a><span>Stitch Fix / Experimentation</span></li>
    </ul>
  </div>
</section>
