---
layout: case-study
title: Food-crisis early warning from bilingual news
permalink: /portfolio/food-crisis-early-warning/
description: An auditable bilingual NLP and geospatial screening system for food-insecurity early warning in the Mashriq.
---

<section class="case-study-hero case-study-hero--policy">
  <p class="eyebrow">Development economics · Humanitarian early warning · Applied NLP</p>
  <h1>Turning bilingual news into an auditable food-crisis verification queue</h1>
  <p class="lead">
    I designed a research prototype that converts English and Arabic news into location-level evidence for food-insecurity monitoring across Iraq, Jordan, Lebanon, Palestine, and Syria. The system helps analysts decide where to investigate—not whether a crisis exists.
  </p>
  <div class="hero__actions">
    <a class="button button--primary" href="/portfolio/">Back to portfolio</a>
  </div>
</section>

<section class="project-metrics" aria-label="Project scope">
  <div><strong>5</strong><span>Mashriq countries</span></div>
  <div><strong>2</strong><span>news languages</span></div>
  <div><strong>167</strong><span>risk-factor phrases</span></div>
  <div><strong>357</strong><span>geographic entities</span></div>
  <div><strong>12</strong><span>risk themes</span></div>
</section>

<section class="content-section case-study-section case-study-section--narrow">
  <p class="eyebrow">The question</p>
  <h2>Can fast-moving news strengthen food-security surveillance when outcome data arrive late?</h2>
  <p>
    Food crises are local, dynamic, and costly to miss. Yet the classifications used to measure food insecurity are often released less frequently than news. Articles can surface drought, price pressure, conflict, displacement, crop failure, and access constraints sooner—but media signals are noisy, unevenly distributed, and easy to over-interpret.
  </p>
  <p>
    The objective was therefore not to label places as “in crisis.” It was to create a transparent screening tool that ranks locations for analyst verification, preserves the evidence behind every flag, and can later be tested against future IPC or Cadre Harmonisé outcomes.
  </p>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Design principles</p>
    <h2>The constraints became part of the methodology</h2>
  </div>
  <div class="case-study-grid">
    <article class="case-study">
      <p class="card-meta">Data scarcity</p>
      <h3>No labels, no false precision</h3>
      <p>With one month of articles and no food-security outcome, a supervised model would not be credible. I built an evidence index and a leakage-safe evaluation plan instead.</p>
    </article>
    <article class="case-study">
      <p class="card-meta">Multilingual evidence</p>
      <h3>English and Arabic stay visible</h3>
      <p>Language-specific retrieval is reported separately before signals are combined, making coverage gaps and translation risk inspectable.</p>
    </article>
    <article class="case-study">
      <p class="card-meta">Operational accountability</p>
      <h3>Every score links back to evidence</h3>
      <p>Analysts can inspect the article, source, language, matched phrase, theme, and location behind a flag before escalating it.</p>
    </article>
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Methodology</p>
    <h2>From raw articles to a policy-facing review queue</h2>
  </div>
  <ol class="method-steps">
    <li>
      <span>01</span>
      <div>
        <h3>Audit and repair the taxonomy</h3>
        <p>I standardized punctuation and encoding, repaired a malformed <em>coup d’état</em> entry that broke the category join, and validated all 167 factors against 12 themes.</p>
      </div>
    </li>
    <li>
      <span>02</span>
      <div>
        <h3>Build a reproducible bilingual seed lexicon</h3>
        <p>The supplied Arabic column was empty. I added one deterministic Arabic seed phrase per factor, normalized diacritics and common spelling variants, and documented the need for native-speaker adjudication and dialect expansion.</p>
      </div>
    </li>
    <li>
      <span>03</span>
      <div>
        <h3>Retrieve phrases without substring leakage</h3>
        <p>A longest-first token n-gram matcher prevents a term such as <em>rise</em> from matching <em>surprise</em> or being double-counted inside <em>price rise</em>. Exact hashes remove duplicate articles.</p>
      </div>
    </li>
    <li>
      <span>04</span>
      <div>
        <h3>Resolve geography conservatively</h3>
        <p>Articles are matched to the supplied country, province, and district aliases. The resolver keeps the most specific unambiguous location and falls back to a shared parent rather than inventing precision.</p>
      </div>
    </li>
    <li>
      <span>05</span>
      <div>
        <h3>Separate signal from confidence</h3>
        <p>A beta-binomial empirical-Bayes model partially pools sparse locations. The signal score summarizes risk prevalence, mention intensity, and thematic breadth; confidence separately reflects article volume, source diversity, and bilingual support.</p>
      </div>
    </li>
    <li>
      <span>06</span>
      <div>
        <h3>Connect prediction to decisions</h3>
        <p>The analysis applies a cost-sensitive prevention framework: a desk review, field verification, and material intervention should each have different thresholds because their costs and effectiveness differ.</p>
      </div>
    </li>
  </ol>
</section>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">Analytical workflow</p>
    <h2>Four steps turn reporting into a review queue</h2>
  </div>
  <ol class="analysis-flow" aria-label="Analysis workflow">
    <li>
      <span class="analysis-flow__number">01</span>
      <span class="analysis-flow__label">Inputs</span>
      <strong>News, risk taxonomy and place names</strong>
    </li>
    <li>
      <span class="analysis-flow__number">02</span>
      <span class="analysis-flow__label">Measure</span>
      <strong>Bilingual phrase and location matching</strong>
    </li>
    <li>
      <span class="analysis-flow__number">03</span>
      <span class="analysis-flow__label">Estimate</span>
      <strong>Partially pooled location-level signals</strong>
    </li>
    <li>
      <span class="analysis-flow__number">04</span>
      <span class="analysis-flow__label">Act</span>
      <strong>Evidence-linked analyst review queue</strong>
    </li>
  </ol>
</section>

<section class="content-section case-study-results">
  <div class="section-heading">
    <p class="eyebrow">Results</p>
    <h2>A complete measurement prototype—with an explicit boundary on what it proves</h2>
  </div>
  <div class="split-section">
    <div>
      <h3>What the implementation delivered</h3>
      <ul class="evidence-list">
        <li>All 167 risk factors retained, categorized, and assigned an Arabic seed phrase.</li>
        <li>Hierarchical matching across 357 countries, provinces, and districts.</li>
        <li>Article-level evidence tables and bilingual retrieval diagnostics.</li>
        <li>A location ranking with uncertainty, source diversity, and language corroboration.</li>
        <li>Robustness tests across multiple empirical-Bayes prior strengths.</li>
        <li>An operational plan for forward validation, calibration, monitoring, and governance.</li>
      </ul>
    </div>
    <aside class="evidence-note">
      <p class="card-meta">Important interpretation</p>
      <h3>The current charts are validation outputs, not crisis estimates</h3>
      <p>The assessment bundle available for this implementation did not include the two news CSVs. I used a clearly labeled synthetic fixture to exercise every pipeline stage and saved the resulting validation outputs. I do not present those examples as empirical findings.</p>
      <p>This is a deliberate research choice: missing data should narrow the claim, not lower the standard of evidence.</p>
    </aside>
  </div>
</section>

<figure class="portfolio-figure">
  <img src="/images/projects/food-crisis-early-warning/lexicon-and-retrieval.png" alt="Two direct-labelled bar charts showing risk-factor taxonomy coverage and phrases retrieved from the validation fixture">
  <figcaption><strong>Retrieval diagnostics.</strong> The left panel audits what the dictionary can detect; the right panel shows what the test corpus triggers. The synthetic fixture validates behavior but supports no geographic conclusion.</figcaption>
</figure>

<figure class="portfolio-figure">
  <img src="/images/projects/food-crisis-early-warning/location-verification-queue.png" alt="Illustrative location verification queue separating signal score from evidence confidence">
  <figcaption><strong>Decision-oriented output.</strong> Signal and confidence are deliberately separate. A high-signal, low-confidence location should trigger verification—not an automated declaration or intervention.</figcaption>
</figure>

<section class="content-section">
  <div class="section-heading">
    <p class="eyebrow">From prototype to forecast</p>
    <h2>How I would test whether news adds value</h2>
  </div>
  <div class="card-grid card-grid--two">
    <article class="feature-card">
      <h3>Build a vintage-correct district-month panel</h3>
      <p>Join future IPC/CH outcomes to lagged news, crisis history, food prices, CHIRPS rainfall, vegetation anomalies, conflict and humanitarian-access events, displacement, population, seasonality, and neighboring-district signals.</p>
    </article>
    <article class="feature-card">
      <h3>Compare nested baselines</h3>
      <p>Evaluate prevalence-only, crisis-history, structured-only, news-only, and combined models. The relevant claim is the incremental performance of news over history and structured early-warning data.</p>
    </article>
    <article class="feature-card">
      <h3>Validate forward in time and across space</h3>
      <p>Use expanding time windows and geographic holdouts. Report precision-recall performance, top-K recall under analyst capacity, Brier score, calibration, and country/language subgroups.</p>
    </article>
    <article class="feature-card">
      <h3>Choose thresholds from policy costs</h3>
      <p>Translate calibrated risk into actions using intervention cost, effectiveness, and the cost of a missed crisis. Monitor source outages, media imbalance, geographic coverage, and score drift after deployment.</p>
    </article>
  </div>
</section>

<section class="content-section split-section">
  <div>
    <p class="eyebrow">Why this work matters</p>
    <h2>Relevance to development institutions</h2>
    <p>The project connects text-as-data, spatial targeting, uncertainty, and decision theory to a real operational problem. It is designed for settings where timeliness matters, labels are scarce, and false confidence can misallocate limited attention or resources.</p>
  </div>
  <div>
    <p class="eyebrow">Capabilities demonstrated</p>
    <div class="tag-cloud">
      <span>Development economics</span>
      <span>Food security</span>
      <span>Multilingual NLP</span>
      <span>Geospatial analytics</span>
      <span>Bayesian shrinkage</span>
      <span>Rare-event forecasting</span>
      <span>Decision theory</span>
      <span>Responsible AI</span>
      <span>Research design</span>
    </div>
  </div>
</section>

<section class="project-cta">
  <div>
    <p class="eyebrow">More work</p>
    <h2>Explore more applied data projects</h2>
    <p>Return to the portfolio for more work at the intersection of development economics, policy, and data science.</p>
  </div>
  <a class="button button--primary" href="/portfolio/">View portfolio</a>
</section>
