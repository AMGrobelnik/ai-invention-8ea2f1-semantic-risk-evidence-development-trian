# SREDT Foundations: GDELT API, GICS/GARP Taxonomies, Venter Diagnostics

## Summary

This research resolves five concrete methodology pillars for the SREDT (Sector-Risk Editorial Development Triangle) experiment. (1) NEWS API: GDELT via BigQuery (table: gdelt-bq.gdeltv2.geg_gcnlapi) is confirmed as the only free option for 2023-2024 European corporate news with structured entity filtering. The entity type field uses Google Cloud NLP values; filter entities.type='ORGANIZATION' to retrieve company mentions by plain-text name. GDELT DOC 2.0 API is limited to 1 year back (as of 2018), insufficient for 2023-2024 historical access. (2) GICS TAXONOMY: Energy sector (code 10) has 1 industry group ('Energy', 1010), 2 industries (Energy Equipment & Services 101010, Oil Gas & Consumable Fuels 101020), and 7 sub-industries (10101010-10102050). Financials sector (code 40) has 3 industry groups (Banks 4010, Financial Services 4020, Insurance 4030), 6 industries, and 18 sub-industries (40101010-40301050). All canonical label strings are enumerated. (3) GARP TAXONOMY: GARP's April 2025 publication confirms 6 Level-1 categories: Operational, Business, Strategic, Reputational, Financial, ESG. GARP does not publish a complete canonical L2 list; the report assembles Level-2 sub-categories from ERM practice and supplements with Basel II / Solvency II operational risk taxonomy (7 L1 categories with Level-2 detail). A recommended set of 20 L3 centroid embedding strings for SREDT is provided. (4) VENTER DIAGNOSTICS: Venter (1998) specifies OLS regression of incremental losses q(w,d+1) on cumulative c(w,d) WITH intercept: q=a+b*c+ε. Factor significance criterion is |b|≥2×SE(b) (p≈4.5%); intercept significance signals additive process. Volume-weighted development factor: f(d)=Σq(w,d+1)/Σc(w,d). The CV<0.3/CV>0.5 thresholds are NOT from Venter's paper — they are CAS actuarial exam convention. BF formula confirmed: C_ultimate=C_observed+E_prior*(1-1/CDF). (5) EDITORIAL LAG: No peer-reviewed study directly measures macro-synthesis commentary lag. Peress (2011) and Engelberg & Parsons establish 1-day micro-event news diffusion. The 3-6 week macro-commentary lag is plausible based on editorial publication cadence (monthly sector newsletters, quarterly analyst cycles) but remains an unverified structural assumption requiring empirical validation in the experiment itself. All findings are documented in research_out.json and research_report.md.

## Research Findings

**1. NEWS API — GDELT BigQuery [1,2,11]:** GDELT via BigQuery is the only free-tier option for 2023-2024 European corporate news with entity filtering. The canonical table is `gdelt-bq.gdeltv2.geg_gcnlapi`. Entity type filter: `entities.type = 'ORGANIZATION'` (using Google Cloud NLP API types [11]). Entity names are plain text (e.g., 'Deutsche Bank'). Date syntax: `date >= "2023-01-01 00:00:00" AND date < "2025-01-01 00:00:00"`. The GDELT DOC 2.0 API is limited to the past 1 year only (since a 2018 update [2]), making BigQuery mandatory for 2023-2024 historical data. BigQuery free tier (1 TB/month) is sufficient for targeted queries.

**2. GICS TAXONOMY [3,12,13]:** Energy sector (10) has ONE industry group — 'Energy' (code 1010) — containing 2 industries (Energy Equipment & Services 101010; Oil, Gas & Consumable Fuels 101020) and 7 sub-industries. Financials sector (40) has THREE industry groups: Banks (4010), Financial Services (4020), Insurance (4030), totalling 6 industries and 18 sub-industries. Complete canonical label strings and 8-digit codes are enumerated in research_out.json. The 2023 GICS update did NOT reclassify renewable energy sub-industries.

**3. GARP RISK TAXONOMY [4,5,8]:** GARP's 2025 publication confirms 6 Level-1 categories: Operational, Business, Strategic, Reputational, Financial, ESG. GARP does not publish a canonical Level-2 list; L2 categories are assembled from ERM practice (Financial→Market/Credit/Liquidity; Operational→Technology/People/Process/Legal; etc.). For Operational Risk, the Basel II/Solvency II taxonomy provides the most authoritative 7-category L2 list [8]. A recommended 20-label L3 centroid embedding string set is provided for SREDT use.

**4. VENTER DIAGNOSTICS [6,7]:** Regression specification: OLS of incremental losses q(w,d+1) on cumulative c(w,d) WITH intercept: `q = a + b*c + ε`. Factor significance criterion: |b| ≥ 2×SE(b) (Venter's exact words, p.814 [6]), corresponding to p≈4.5%. Intercept significance (|a|≥2×SE(a)) signals additive process favoring 'factor plus constant' method over chain-ladder. Volume-weighted development factor: `f(d) = Σ_w q(w,d+1) / Σ_w c(w,d)`. BF formula: `C_ultimate = C_observed + E_prior × (1 - 1/F(d))` [7]. **CRITICAL:** CV<0.3/CV>0.5 thresholds are NOT from Venter's paper — they are CAS actuarial exam convention. No single authoritative published source for these specific thresholds was located [6,14].

**5. EDITORIAL LAG [9,10]:** Academic literature confirms 1-day micro-level news diffusion (Peress 2011 [9]; Engelberg & Parsons [10]), but no study directly measures macro-synthesis commentary lag. The 3-6 week lag hypothesis is consistent with editorial publication cadence (monthly sector newsletters, quarterly analyst cycles) but is NOT empirically documented. It should be treated as a motivated structural assumption and validated empirically within the SREDT experiment.

## Sources

[1] [Querying The New Global Entity Graph (GEG) Datasets In BigQuery: Example Queries](https://blog.gdeltproject.org/querying-the-new-global-entity-graph-geg-datasets-in-bigquery-example-queries/) — Provides exact BigQuery SQL syntax for querying geg_gcnlapi, including entity.type filter, entity.name format (plain text), date range syntax, and the unnest() operator required for REPEATED fields.

[2] [DOC & GEO 2.0 API Updates: Full Year Searching And More! (GDELT, 2018)](https://blog.gdeltproject.org/doc-geo-2-0-api-updates-full-year-searching-and-more/) — Confirms GDELT DOC 2.0 API maximum search horizon is 1 year. This makes it unsuitable for 2023-2024 historical access, mandating BigQuery.

[3] [Global Industry Classification Standard - Wikipedia](https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard) — Complete and current GICS classification table with all sector, industry group, industry, and sub-industry codes and labels. Source for all GICS taxonomy data.

[4] [ERM: The Importance of Building a Risk Taxonomy (GARP, Boultwood & Switzer, April 2025)](https://www.garp.org/risk-intelligence/culture-governance/erm-importance-building-250417) — Most recent GARP publication confirming 6 Level-1 risk categories: operational, business, strategic, reputational, financial, ESG.

[5] [How to Develop an Enterprise Risk Taxonomy (GARP, Boultwood, July 2021)](https://www.garp.org/risk-intelligence/culture-governance/how-to-develop-an-enterprise-risk-taxonomy) — Describes 4-level ERM hierarchy; confirms Level-2 categories are firm-specific and not published as a single canonical GARP list. Illustrative L2 example: 'budget adequacy and pressures' under Business Risk.

[6] [Testing the Assumptions of Age-to-Age Factors — Venter (1998), PCAS LXXXV](https://www.casact.org/sites/default/files/2021-03/7_Venter_Factors.pdf) — Primary source for Venter diagnostics. Specifies incremental-on-cumulative regression with intercept; factor significance criterion |b|≥2×SE(b); chain-ladder vs. BF comparison via adjusted SSE. CV thresholds NOT present in this paper.

[7] [Bornhuetter-Ferguson Loss Reserving in R: A Full Guide (MetricGate, 2026)](https://metricgate.com/blogs/loss-development-bornhuetter-ferguson-r/) — Confirms BF formula: C_ultimate = C_observed + E[U]*(1-1/CDF). Reserve = E[U]*(1-percent_developed). Volume-weighted development factors are the standard chain-ladder approach.

[8] [Insurance Operational Risk Taxonomy: Basel II/Solvency II — Patel (CAS 2010)](https://www.casact.org/sites/default/files/old/reinsure_2010_handouts_cs14-patelappendix.pdf) — Full Basel II/Solvency II operational risk taxonomy with 7 L1 categories and their L2 sub-categories. Most authoritative published operational risk L2 taxonomy available.

[9] [The Media and the Diffusion of Information in Financial Markets — Peress (2011/2014, Journal of Finance)](https://www.eief.it/files/2012/04/joel-peress.pdf) — Shows media propagates news from previous day (1-day lag). Trading volume falls 14% on newspaper strike days. Establishes 1-day lower bound for news diffusion; no direct evidence on macro-synthesis commentary lag.

[10] [The Causal Impact of Media in Financial Markets — Engelberg & Parsons](https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/MEDIA.pdf) — Confirms local media coverage of earnings events predicts local trading within days. Supports 1-7 day micro-event reaction lag; macro-synthesis lag not measured.

[11] [Google Cloud Natural Language API — Entity Type Reference](https://docs.cloud.google.com/natural-language/docs/reference/rest/v1/Entity) — Confirms entity type enum values used in GDELT GEG: UNKNOWN, PERSON, LOCATION, ORGANIZATION, EVENT, WORK_OF_ART, CONSUMER_GOOD, OTHER, PHONE_NUMBER, ADDRESS, DATE, NUMBER, PRICE.

[12] [Global Industry Classification Standard (GICS) - Energy Sector — Lexchart](https://lexchart.com/global-industry-classification-standard-gics-energy-sector/) — Detailed Energy sector GICS structure breakdown confirming industry groups and sub-industry hierarchy.

[13] [GICS — Global Industry Classification Standard — MSCI (Official)](https://www.msci.com/indexes/index-resources/gics) — Official MSCI GICS page confirming current structure: 11 sectors, 25 industry groups, 74 industries, 163 sub-industries.

[14] [Study Manual for CAS Exam 7 (5th Ed.) — ACTEX Learning 2026](https://www.actexlearning.com/samples/7C-ACTEX-5E-SM-E_Sample.pdf) — CAS Exam 7 study material covering Venter (1998). No CV 0.3/0.5 thresholds found in sample — supporting conclusion that these are practitioner convention without a definitive canonical source.

## Follow-up Questions

- For GDELT entity matching: company names in geg_gcnlapi appear in many spelling variants. What fraction of major European DAX/CAC40/AEX/FTSE100 companies have confirmed MIDs (Machine IDs) in the GEG for canonicalization, and what is the recommended fallback when MID is NULL?
- The CV < 0.3 / CV > 0.5 thresholds for chain-ladder vs. BF selection are CAS convention without a definitive primary source. Does Friedland (2010) 'Estimating Unpaid Claims Using Basic Techniques' Chapter 7 specify these thresholds? A targeted grep of that chapter would either confirm or definitively rule out this attribution.
- The 3-6 week editorial lag is an unverified structural assumption. Can the SREDT experiment empirically measure this lag distribution as a pre-analysis step using GDELT cross-correlations between GICS sub-industry event frequency and GARP risk category mention frequency, and what is the minimum required time series length to estimate the lag reliably?

---
*Generated by AI Inventor Pipeline*
