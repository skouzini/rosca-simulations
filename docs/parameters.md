# Simulation parameters: sources and recommendations

Research date: 2026-11-19. Covers every default in `rosca/params.py`. "Assumption" means no primary source was found or the value is a design choice.

## Summary of recommended changes

The biggest corrections are on the **payday side**. Utah's own regulator reports an **average loan of $465** (not Pew's national $375), an **average APR of 469%** (≈ **$18 per $100** per two weeks, not $21), and paid-off Utah loans carrying interest for an **average of 23 days**. That last figure means a per-term rollover probability near **0.4**, not 0.8, if `renew_prob` models rolling over the *same* loan. The CFPB's ">80% renewed within 14 days" also counts brand-new loans taken right after repayment. On the **shock side**, the Fed's SHED reports "major" unexpected expenses, mostly costing **$1,000–1,999**. About 59% of adults had at least one in 2025, which fits roughly **1–1.3 major shocks per year**. The current setting of 2.5 shocks a year at a $400 median uses the $400 figure as if it were an observed expense size. It is not: $400 is only the hypothetical amount in the survey question. **Job loss** at 5% a year is low next to JOLTS layoff rates of 1.0–1.4% *per month*. **Rent** should move from $1,350 to the FY2026 HUD Fair Market Rent ($1,456 for a one-bedroom, $1,747 for a two-bedroom). Wages check out: Salt Lake MSA medians for typical hourly jobs are $14–22, so the $17 default holds. One more flag: at $17/hr × 30–32 hrs, a single earner cannot cover FMR rent plus any realistic budget for essentials. The baseline is close to insolvent by construction. Read "Budget consistency warning" before changing rent or essentials.

## Parameter table

Legend: **⚠ CHANGE** means the recommended value differs from the current default.

### Population (`PopulationParams`)

| Parameter | Current | Recommended | Source | Conf. | Note |
|---|---|---|---|---|---|
| `cohort_size` | 12 | 12 (range 6–20) | [MAF Lending Circles page (2026)](https://www.missionassetfund.org/lending-circles/): circles have 6–12 people. [CRS SILC brief (2012)](https://www.crs.org/sites/default/files/2025-06/group-performance-in-fee-for-service-savings-groups.pdf): savings groups average 19–27 members. | medium | Keep. Design choice that sits inside the observed range. |
| `wage_median` | 17.0 | 17.0 (range 14–22) | [BLS OEWS May 2025, Salt Lake City MSA 41620](https://www.bls.gov/oes/current/oes_41620.htm), via the BLS API | high | Verified. Employment-weighted median across the 14 hourly occupations below is **$17.91**, or **$16.43** without customer service reps and hand laborers. See the wage table. |
| `wage_sigma` | 0.20 | 0.20 (0.15–0.25) | Same OEWS data | medium | The 25th-percentile to median ratio in these occupations runs 0.77–0.95, which implies a lognormal sigma of about 0.08–0.38. 0.20 sits in the middle. |
| `hours_mean` | 32 | **⚠ 30** (range 24–34) | [BLS CES, Oct 2026](https://data.bls.gov/timeseries/CES0500000007): avg weekly hours of production/nonsupervisory employees. Total private 33.8; retail 30.7 ([CES4200000007](https://data.bls.gov/timeseries/CES4200000007)); food services 23.8 ([CES7072200007](https://data.bls.gov/timeseries/CES7072200007), Sep 2026). Salt Lake MSA all-employee private average is 35.0 ([SMU49416200500000002](https://data.bls.gov/timeseries/SMU49416200500000002), Sep 2026). | medium | The target occupations lean toward retail and food service, so 30 fits better than 32. |
| `hours_sd_frac` | 0.15 | 0.15 weekly, but see note | [JPMC Institute, *Weathering Volatility 2.0* (2019)](https://www.jpmorganchase.com/institute/all-topics/financial-health-wealth-creation/report-weathering-volatility-2-a-monthly-stress-test-to-guide-saving): median monthly income CV is about 0.38 and the typical family sees a 36% month-to-month change. [SHED 2025 (2026)](https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-income-and-expenses.htm): 30% of adults had income that varied at least occasionally. | low | **Mismatch flag.** Independent weekly noise with sd 0.15 gives a monthly CV of only about 0.07. JPMC's 0.38 covers all income (tax refunds, bonuses, job changes), so it should be matched by noise, hours-cut shocks and job-loss shocks together. Use it as a calibration target, not as this parameter. |
| `rent_median` | 1350 | **⚠ 1,456** (1BR); 1,747 for 2-bedroom families | [HUD FY2026 FMR, Salt Lake City UT HUD Metro FMR Area (Salt Lake County)](https://www.huduser.gov/portal/datasets/fmr/fmrs/FY2026_code/2026summary.odn?&year=2026&fmrtype=Final&selection_type=county&fips=4903599999): 0BR $1,259, 1BR $1,456, 2BR $1,747, 3BR $2,333. FY2025: 1BR $1,453, 2BR $1,748. | high (the FMR values) / medium (which bedroom size) | FMRs are HUD's 40th-percentile gross rent, which includes utilities. Single parents and young families mostly need a 2BR. No FY2027 FMR page was available when checked (404). |
| `rent_sigma` | 0.15 | 0.15 | Assumption | low | The spread between 0BR and 2BR FMRs (±16%) roughly fits this. |
| `essentials_monthly` | 900 | 900 kept, **flagged low** | [BLS Consumer Expenditure Survey 2025](https://www.bls.gov/cex/tables.htm), lowest income quintile (pre-tax income $18,125): total spending $37,935, shelter $10,369, so non-shelter spending is about **$2,300/mo** (series CXUTOTALEXPLB0102M, CXUSHELTERLB0102M). [MIT Living Wage, Salt Lake County (Feb 2026)](https://livingwage.mit.edu/counties/49035): non-housing basics are about **$1,900/mo** for 1 adult, and about **$3,800/mo** for 1 adult + 1 child (childcare $10,986/yr). | low | Both benchmarks are 2–4× the default. Raising it without also raising income makes every agent insolvent. See the budget consistency warning. |
| `buffer_median` | 300 | 300 (assumption) | [SHED 2025 (2026)](https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-savings-investments.htm): 63% would cover a $400 expense with cash or its equivalent; 12% could not pay it by any means; only 21% of adults under $25k have 3 months of expenses saved. JPMC 2019: 46% of families lack the 2.6 weeks of take-home pay needed for a typical spending spike. | low | No primary source gives the median liquid balance for this group. 300 is consistent with "cannot cover $400 from cash", which describes the target population better than the national 37%. |
| `buffer_sigma` | 1.0 | 1.0 | Assumption | low | — |
| `payday_propensity` | 0.7 | **⚠ calibrate; 0.7 only if the cohort is defined as existing payday users**, otherwise 0.1–0.3 | [SHED 2025](https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-savings-investments.htm): only 2% of all adults (about 5% of the 37% without cash) would use a payday loan, deposit advance or overdraft for a $400 expense. [Pew 2012](https://www.pew.org/-/media/legacy/uploadedfiles/pcs_assets/2012/pewpaydaylendingreportpdf.pdf): 5.5% of adults used a payday loan in the prior 5 years; renters use them far more than homeowners. | low | Present bias justifies a value above the national average, but 0.7 is not sourced. Treat it as the main free knob for matching the calibration targets. |
| `strategic_default_prob` | 0.01 | 0.01 (range 0.007–0.06) | MAF self-reported 0.7% default ([Next City, Sept 2016](https://nextcity.org/daily/entry/lending-circles-predatory-loans)); "1%" (Marketplace, 2013); 99.02% repayment ([JPMorgan Chase story](https://www.jpmorganchase.com/newsroom/stories/expanding-financial-access), sponsor-published, undated). Upper bound: in Gugerty's Kenya ROSCAs, 6% of members left during the last round (as summarized in [Basu 2008](https://mpra.ub.uni-muenchen.de/20440/1/MPRA_paper_20440.pdf)). | medium | Keep. MAF's figure is self-reported, unaudited and comes with MAF guaranteeing the loans. Gugerty's 6% is *exit*, not default. No primary MAF document with a default rate was found. |

#### Salt Lake City MSA wages, BLS OEWS May 2025 (hourly)

| Occupation (SOC) | Employment | 10th pct | 25th pct | Median |
|---|---|---|---|---|
| All occupations | 826,210 | 15.26 | 18.78 | 26.56 |
| Fast food & counter workers (35-3023) | 19,590 | 10.85 | 13.05 | **13.98** |
| Waiters & waitresses (35-3031) | 7,670 | 10.26 | 10.73 | 15.54 |
| Food preparation workers (35-2021) | 4,880 | — | 14.73 | 16.04 |
| Cooks, restaurant (35-2014) | 4,600 | 13.94 | 16.40 | 17.65 |
| Cashiers (41-2011) | 9,680 | 12.74 | 14.28 | **16.06** |
| Retail salespersons (41-2031) | 20,330 | 13.32 | 14.20 | **16.55** |
| Stockers & order fillers (53-7065) | 11,650 | — | 16.48 | 18.04 |
| Laborers & material movers, hand (53-7062) | 15,630 | — | 18.13 | **21.25** |
| Home health & personal care aides (31-1120) | 6,330 | — | 16.91 | **17.85** |
| Nursing assistants (31-1131) | 5,160 | — | 18.60 | 19.48 |
| Janitors & cleaners (37-2011) | 10,610 | — | 13.73 | 16.81 |
| Maids & housekeeping (37-2012) | 3,650 | — | 16.97 | 17.82 |
| Childcare workers (39-9011) | 2,010 | — | 14.26 | 16.25 |
| Customer service reps (43-4051) | 26,620 | — | 18.43 | 21.83 |

Pulled from the BLS Public Data API, series `OEUM004162000000{SOC}{08=median, 07=p25, 06=p10, 01=employment}`. bls.gov blocks scripted page fetches (HTTP 403), so the HTML table was not read directly.

### Shocks (`ShockParams`)

| Parameter | Current | Recommended | Source | Conf. | Note |
|---|---|---|---|---|---|
| `expense_rate_per_year` | 2.5 | **⚠ 1.2** (range 0.9–1.5) for "major" shocks | [SHED 2025 (2026), Economic Hardships](https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-economic-hardships.htm): 59% of adults had ≥1 major unexpected expense in the prior 12 months (47% for income <$25k, 61% for $25–50k). By type: vehicle 30%, house/appliance 22%, medical 21%, phone/computer 18%, legal/taxes/fines 10%, other 6%, childcare increase 3%. | medium | P(≥1) = 0.59 implies a Poisson λ of about 0.9. The type shares add to 110%, so the true count is at least about 1.1. If the model also needs small shocks under about $300, treat them separately or fold them into `essentials`. |
| `expense_median` | 400 | **⚠ 1,000** | Same source, Figure 23: median cost is **$1,000–1,999** for vehicle, house/appliance and medical; **$500–999** for phone/computer; **$2,000–4,999** for legal/taxes/fines. | medium | $400 is the *hypothetical* amount in the SHED question, not an observed shock size. $1,000 is the low end of the dominant bin, chosen conservatively. |
| `expense_sigma` | 1.256 | **⚠ 0.8** (assumption) | SHED publishes binned medians only | low | 0.8 gives p10 of about $360 and p90 of about $2,800, which covers the phone-to-legal range. Expected annual major-shock spending is then about $1,650, against about $2,200 under the current defaults. |
| `job_loss_rate_per_year` | 0.05 | **⚠ 0.12** (range 0.10–0.17) | [BLS JOLTS](https://data.bls.gov/timeseries/JTS000000000000000LDR) layoffs & discharges rate, Sep 2026: total nonfarm 1.0%/mo; leisure & hospitality 1.4%/mo ([JTS700000000000000LDR](https://data.bls.gov/timeseries/JTS700000000000000LDR)); retail 1.0%/mo ([JTS440000000000000LDR](https://data.bls.gov/timeseries/JTS440000000000000LDR)). | medium | 1.0–1.4%/mo works out to about 12–17%/yr. Some laid-off workers find new work right away, so 0.12 is a conservative central value. |
| `job_loss_weeks_mean` | 8 | **⚠ 10** (range 8–12) | [BLS CPS](https://data.bls.gov/timeseries/LNS13008276): median unemployment duration 11.9 weeks (Oct 2026); mean 27.1 weeks ([LNS13008275](https://data.bls.gov/timeseries/LNS13008275)), pulled up by the long-term unemployed. | medium | CPS covers all unemployed people. Low-wage hourly workers usually find work faster, so use 10 rather than 12. |

### Payday (`PaydayParams`)

| Parameter | Current | Recommended | Source | Conf. | Note |
|---|---|---|---|---|---|
| `loan_size` | 375 | **⚠ 465** | [Utah DFI 46th Annual Report, Oct 2026](https://dfi.utah.gov/wp-content/uploads/2026/10/Annual.pdf), p.136 "Deferred Deposit Lenders – Aggregate Information", calendar 2025, 18 lenders reporting: average loan **$465**. Prior reports: $504 (CY2024, [Oct 2025 report](https://dfi.utah.gov/wp-content/uploads/2025/10/Annual.pdf)) and $487 (CY2023, [Oct 2024 report](https://dfi.utah.gov/wp-content/uploads/2024/10/Annual.pdf)). | high | Pew's $375 (2012) is national and 14 years old. Use $465 for Utah, with a $465–505 range. |
| `fee_per_100` | 21.0 | **⚠ 18.0** | Utah DFI Oct 2026: average APR **469.07%** (range 16.57%–1,642%). Earlier reports: 498.32% (CY2024) and 567.27% (CY2023). | high | 469.07 / 26 = $18.04 per $100 per 2-week term. The old comment's "554%, DFI 2020" is out of date. No rate cap was found in Utah Code 7-23-401 or 7-23-403. |
| `term_weeks` | 2 | 2 | [CFPB 2013 white paper](https://files.consumerfinance.gov/f/201304_cfpb_payday-dap-whitepaper.pdf): median term 14 days, mean 18.3 days. CFPB 2014: 47% of borrowers paid biweekly, 14% weekly, 9% semi-monthly, 30% monthly. | high | Keep. |
| `max_weeks` | 10 | 10 | [Utah Code 7-23-401(4)](https://le.utah.gov/xcode/Title7/Chapter23/7-23-S401.html) (effective 7/1/2024): no additional interest on a balance outstanding 10 weeks after the loan is executed; no rollover that requires repayment more than 10 weeks after execution; rollovers only at the borrower's request. | high | Verified. Same statute: the borrower can rescind with no charge by 5 p.m. the next business day, and can prepay in increments of at least $5. |
| `epp_weeks` | 8 | **⚠ 9** (statutory minimum, ≥ 60 days); 13 if entered after default | [Utah Code 7-23-403](https://le.utah.gov/xcode/Title7/Chapter23/7-23-S403.html) (effective 7/1/2016): available at least once per 12 months, and mandatory on request after 10 weeks of interest; **at least 4 equal payments** over at least the greater of **90 days after default or 60 days after entering the plan**; **no interest or fees**; on default during the plan, the lender may charge one fee of **up to $20** and accelerate the balance. | high | 60 days ≈ 8.6 weeks, so 8 weeks is slightly shorter than the law allows. The DFI average plan balance is **$268** (CY2025). **3.30%** of borrowers entered a plan in CY2025 (3.21% in CY2024, 4.54% in CY2023). |
| `renew_prob` | 0.8 | **⚠ about 0.4 per 2-week term if this means rolling over the same loan**; keep 0.8 only if it means "any new loan within 14 days" | CFPB 2014 [Data Point: Payday Lending](https://files.consumerfinance.gov/f/201403_cfpb_report_payday-lending.pdf): >80% of loans are rolled over **or followed by another loan within 14 days**. Utah DFI Oct 2026: loans paid in full carried interest for an average of **23 days**. | medium | With 2-week terms and the 10-week cap, mean days outstanding = 14·Σ_{k=0..4} p^k. 23 days gives p ≈ 0.40; p = 0.8 would give about 47 days. The CFPB figure comes from multi-state data and includes back-to-back new loans, which Utah counts as separate loans. |
| `default_prob_when_unable` | 0.25 | 0.25, calibrate | CFPB 2014: over an 11-month window, **20%** of new borrowers defaulted, 64% became renewers and 15% repaid; 4% of new sequences default on the first loan. Utah DFI: civil action filed on **1.15%** of loans (CY2025); 0.5% in CY2024; 9.34% in CY2023. | medium | Use the CFPB 20% (borrower-level, per year) as the target. The Utah civil-action share swings too much between years to use and is a lower bound on default, since not every default is sued. |

### Pool (`PoolParams`)

Most of these are **design levers** for the structures being compared, not empirical quantities. Benchmarks are listed where they exist.

| Parameter | Current | Recommended | Source / benchmark | Conf. | Note |
|---|---|---|---|---|---|
| `contribution_weekly` | 50 | 50, also test 25 | [MAF (2026)](https://www.missionassetfund.org/lending-circles/): typical payments $50–200/month (about $12–46/week); loans $300–2,400. | medium | $50/wk is about 9% of gross pay at $17 × 32h, at the top of MAF's range. |
| `round_weeks` | 2 | 2 | CFPB 2014 pay frequency (biweekly is the most common at 47%) | medium | Keep. Matches the payday cycle. |
| `miss_policy` | demote | design choice | — | — | — |
| `expel_after_misses` | 2 | design choice | — | — | — |
| `collapse_min_frac` | 0.5 | design choice; stress benchmark below | [McNabb, LeMay-Boucher & Bonan (2019), EJDR 31:1389–1415](https://ideas.repec.org/a/pal/eurjdr/v31y2019i5d10.1057_s41287-019-00215-5.html): over 2 years, 1 in 3 informal ROSCAs in urban Benin had enforcement problems and two-thirds of those collapsed (about 22% of groups in 2 years). | medium (abstract only; full text not read) | Use this as the collapse benchmark for an *unguaranteed* rigid ROSCA. |
| `seed_liquidity` | 0 | design choice | — | — | MAF guarantees its circles with its own capital, so the low MAF default rate comes with that backstop. |
| `reserve_frac` | 0.20 | design choice | — | — | — |
| `target_reserve_pots` | 1.0 | design choice | — | — | — |
| `draw_cap` | 500 | 500 | Close to the Utah average payday loan ($465); NCUA PAL I minimum is $200 | medium | Keep. Sized to replace one typical payday loan. |
| `draw_repay_weeks` | 8 | 8–9 | Utah EPP minimum is about 8.6 weeks; PAL minimum maturity is 1 month | medium | Matching the EPP (9) makes the comparison with the statutory payday off-ramp cleaner. |
| `payout_floor` | 0.7 | design choice | — | — | — |
| `staking_yield_apr` | 0.04 | 0.04 | [U.S. Treasury daily bill rates, 2026-11-19](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2026), coupon-equivalent: 4-wk **4.12%**, 13-wk **4.26%**, 26-wk 4.47%, 52-wk 4.60% | high (the rate on that date) | Keep 4% as a value net of sweep and MMF fees, about 25 bp under the 13-week bill. Rates move; re-check before presenting. |
| `stake_prob` | 0.5 | assumption | — | low | — |
| `stake_cash_threshold` | 500 | assumption | — | low | — |

### Economics (`EconParams`)

| Parameter | Current | Recommended | Source | Conf. | Note |
|---|---|---|---|---|---|
| `tbill_yield` | 0.04 | 0.04 (gross 13-wk = 4.26% on 2026-11-19) | U.S. Treasury daily bill rates (above) | high (on that date) | You could use 0.0426 gross and model fees in `opex`. |
| `pepm_fee` | 3.0 | assumption ($2–4) | No public benchmark found in this pass | low | Comes from the business draft. Label it as an assumption in presentations. |
| `member_fee_monthly` | 0.0 | assumption ($0–5) | MAF circles are zero-interest, zero-fee to members (Next City 2016) | low | — |
| `opex_per_member_month` | 4.0 | assumption | Not researched / no source found | low | — |

### Budget consistency warning

Gross weekly pay at the defaults is $17 × 32 = $544, or about $2,356/month. That is **before** FICA and income tax (FICA alone is 7.65%). Rent ($1,350) plus essentials ($900) already comes to $2,250. With FY2026 FMR rent ($1,456 1BR / $1,747 2BR) and realistic essentials (CE or MIT: $1,900–2,300+), a single earner at $17/hr runs a structural deficit of $800–2,000 per month. Real households close that gap with a second earner, roommates, SNAP/EITC, or a rent subsidy. Before adopting the higher rent or essentials values, choose one of these:

1. Model household income (for example, 1.3–1.6 earners for young families).
2. Keep rent and essentials scaled to produce a target rent-to-income ratio (cost-burdened, about 40–50%).
3. Keep the current pair, label it explicitly as "post-subsidy net budget", and report it as an assumption.

Otherwise the payday-only baseline produces chronic borrowing that comes from the budget arithmetic, not from shocks. Pew (2012) does find that **69%** of first-time payday borrowers used the loan for **recurring expenses** and only **16%** for an unexpected emergency, so *some* structural shortfall is realistic.

## Calibration targets

The payday-only (`no_pool`) baseline should reproduce these. They are listed in priority order; Utah-specific targets come first.

| # | Target | Value | Source |
|---|---|---|---|
| 1 | Average payday loan amount | $465 (CY2025; $487–504 the prior two years) | Utah DFI Oct 2026 report, p.136 |
| 2 | Average days a paid-in-full loan carries interest | 23 days (22–24 across 3 years) | Utah DFI 2024–2026 reports |
| 3 | Average APR | 469% (2026 report); 498%, 567% in the prior two | Utah DFI |
| 4 | Share of borrowers entering an extended payment plan per year | 3.3% (3.2–4.5%) | Utah DFI |
| 5 | Average extended-payment-plan balance | $268 | Utah DFI CY2025 |
| 6 | Share of new borrowers who default within about 11 months | about 20% | CFPB 2014 Data Point |
| 7 | Borrower types over 11 months | 64% renewers / 20% defaulters / 15% repayers | CFPB 2014 |
| 8 | Loans per borrower per year | Pew: 8 (mean); CFPB 2013: median 10 | Pew 2012; CFPB 2013 |
| 9 | Fees per borrower per year | Pew: $520 interest; CFPB 2013: median $458 | Pew 2012; CFPB 2013 |
| 10 | Time in debt per year | Pew: 5 months; CFPB 2013: median 199 days (55% of the year) | Pew 2012; CFPB 2013 |
| 11 | Share of payday use for recurring vs. unexpected expenses | 69% / 16% | Pew 2012 |
| 12 | Monthly income CV (all sources, median household) | about 0.38 | JPMC 2019 |
| 13 | Share with ≥1 major unexpected expense per year | 59% overall; 47–61% for incomes under $50k | SHED 2025 |

How to read 8–10 in Utah: these figures are national and come from 2010–2012 data, mostly from states without a 10-week cap, so read them as upper bounds. Utah's 10-week cap on a single loan does not stop a borrower from taking out a new loan later. Treat targets 1–5 as hard targets and 6–11 as ranges.

What Utah DFI no longer reports: the current aggregate table (7-23-503(2)(a)) has **no total loan count and no borrower count**, so no Utah *share* of loans reaching 10 weeks can be computed. The table does give counts. CY2025: **4,442** loans carried to the 10-week maximum ($2.87M) and **4,443** not paid in full at 10 weeks ($2.84M). CY2024: 4,771 / 4,586. CY2023: 5,405 / 5,232. Rescissions: 621 loans ($476k) in CY2025. Civil actions: 1.15% of loans; 49% of those had no payments, 38% one payment.

## Comparison products

**NCUA Payday Alternative Loans** ([12 CFR 701.21(c)(7)(iii)–(iv)](https://www.ecfr.gov/current/title-12/chapter-VII/subchapter-A/part-701/section-701.21), text pulled via the eCFR API, 2026-11-01 version):

| | PAL I | PAL II |
|---|---|---|
| Amount | $200–$1,000 | up to $2,000 (no minimum) |
| Term | 1–6 months | 1–12 months |
| Max rate | general ceiling + 1,000 bp = **28%** while the ceiling is 18% | same |
| Application fee | ≤ $20, reflecting actual cost | ≤ $20 |
| Membership wait | ≥ 1 month | none |
| Other | fully amortizing; no rollovers (fee-free extensions within the max term allowed); max 3 PALs per rolling 6 months, 1 at a time; aggregate ≤ 20% of net worth | same, plus no overdraft/NSF fees tied to the PAL |

The 18% ceiling (so 28% for PALs) was extended on 2026-02-06 through **September 10, 2027** ([NCUA press release](https://ncua.gov/newsroom/press-release/2026/ncua-board-extends-loan-interest-rate-ceiling)). Example cost: a $465 PAL at 28% APR repaid over 3 months costs about $22 in interest plus up to $20 in fees. A Utah payday loan of the same size at $18/$100 costs $84 per two-week term.

**Mission Asset Fund Lending Circles** (zero-interest, zero-fee, credit-reported):
- Circles of 6–12 people; loans $300–2,400; payments $50–200/month ([MAF, 2026](https://www.missionassetfund.org/lending-circles/)).
- Reported default rate is 0.7% ([Next City, 2016](https://nextcity.org/daily/entry/lending-circles-predatory-loans)), self-reported by MAF. As of 2016: 4,079 borrowers, $5M in loans, 53 partners in 17 states plus DC.
- Credit outcomes ([Reyes & Lopez, Boston Fed *Communities & Banking*, Dec 2015](https://www.bostonfed.org/publications/communities-and-banking/2016/winter/social-lending-improving-credit-for-the-underbanked.aspx)): average score went from 435 to 603 (+168) for treatment vs. 456 to 497 for controls over about 10 months, from a quasi-experiment with n = 209 treated and 138 controls, 2011–2013. The underlying study is Reyes, Lopez, Phillips & Schroeder (2013), *Building Credit for the Underbanked*, Cesar E. Chavez Institute, SF State. It was not retrieved.
- MAF guarantees the loans with its own capital, which is a different risk structure from an unbacked ROSCA.

**Other savings-group evidence:**
- **CRS SILC (Kenya, Tanzania, Uganda; 2012):** dropout within a cycle was 0.0–0.2% across 1,996 groups; return on savings was 18–27% per cycle ([CRS research brief](https://www.crs.org/sites/default/files/2025-06/group-performance-in-fee-for-service-savings-groups.pdf)).
- **CARE *VSLA by the Numbers* (2022):** return on savings was about 50% (±10%) during the supported cycle and about 35% after phase-out; about 81% of active VSLAs kept saving through COVID lockdowns ([CARE](https://www.care.org/wp-content/uploads/2022/11/VSLA-by-the-Numbers-Report_2022.pdf)). The report has no loan-repayment or portfolio-at-risk figure.
- **Benin informal ROSCAs:** about 22% of groups collapsed over 2 years (McNabb et al. 2019).
- **Kenya:** 6% of members exited per round (Gugerty, as summarized by Basu 2008).
- **US tandas:** no quantitative default or dropout study was found. The literature found was qualitative (for example, interviews with 12 Mexican-American participants). This is a real evidence gap and should be stated as one.
- **Besley, Coate & Loury (1993):** theory only, with no empirical default rates. It establishes that ROSCAs raise welfare by moving lumpy purchases earlier, and compares random and bidding ROSCAs.

## Sources

1. Utah Department of Financial Institutions. *Forty-Sixth Annual and Eightieth Report* (period July 1, 2025–June 30, 2026; payday data for calendar 2025), Oct 2026, pp. 132–137. https://dfi.utah.gov/wp-content/uploads/2026/10/Annual.pdf
2. Utah DFI. *Annual Report*, Oct 2025 (CY2024 payday data). https://dfi.utah.gov/wp-content/uploads/2025/10/Annual.pdf
3. Utah DFI. *Annual Report*, Oct 2024 (CY2023 payday data). https://dfi.utah.gov/wp-content/uploads/2024/10/Annual.pdf. Note: the older `.../uploads/sites/29/YYYY/10/Annual.pdf` URLs found by search now return 404; the links above are current, found via https://dfi.utah.gov/general-information/publications/.
4. Utah Code §7-23-401, Operational requirements for deferred deposit loans (eff. 7/1/2024). https://le.utah.gov/xcode/Title7/Chapter23/7-23-S401.html
5. Utah Code §7-23-403, Extended payment plan (eff. 7/1/2016). https://le.utah.gov/xcode/Title7/Chapter23/7-23-S403.html
6. U.S. Bureau of Labor Statistics. Occupational Employment and Wage Statistics, May 2025, Salt Lake City MSA (41620). https://www.bls.gov/oes/current/oes_41620.htm (values from the BLS Public Data API, series OEUM0041620…)
7. BLS. Current Employment Statistics (CES) average weekly hours, series CES0500000007, CES4200000007, CES7072200007, SMU49416200500000002, retrieved 2026-11-19. https://data.bls.gov/timeseries/CES0500000007
8. BLS. Job Openings and Labor Turnover Survey, layoffs & discharges rate, series JTS000000000000000LDR, JTS700000000000000LDR, JTS440000000000000LDR (Sep 2026). https://data.bls.gov/timeseries/JTS000000000000000LDR
9. BLS. Current Population Survey, median and mean duration of unemployment, series LNS13008276, LNS13008275 (Oct 2026). https://data.bls.gov/timeseries/LNS13008276
10. BLS. Consumer Expenditure Survey 2025, by income quintile, series CXUTOTALEXPLB0102M, CXUSHELTERLB0102M, CXUINCBEFTXLB0102M. https://www.bls.gov/cex/tables.htm
11. U.S. HUD. FY2026 Fair Market Rent Documentation System, Salt Lake City UT HUD Metro FMR Area. https://www.huduser.gov/portal/datasets/fmr/fmrs/FY2026_code/2026summary.odn?&year=2026&fmrtype=Final&selection_type=county&fips=4903599999 (FY2025: same URL with 2025)
12. MIT Living Wage Calculator, Salt Lake County, UT (updated Feb 15, 2026). https://livingwage.mit.edu/counties/49035
13. Board of Governors of the Federal Reserve System. *Economic Well-Being of U.S. Households in 2025* (SHED; fielded Oct 2025, published May 13, 2026). Sections: Economic Hardships https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-economic-hardships.htm ; Savings and Investments https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-savings-investments.htm ; Income and Expenses https://www.federalreserve.gov/publications/2026-economic-well-being-of-us-households-in-2025-income-and-expenses.htm ; press release https://www.federalreserve.gov/newsevents/pressreleases/other20260513a.htm. The 2024 survey (published May 2025) also reported 63% for the $400 question.
14. JPMorgan Chase Institute. *Weathering Volatility 2.0: A Monthly Stress Test to Guide Savings*, Oct 23, 2019. https://www.jpmorganchase.com/institute/all-topics/financial-health-wealth-creation/report-weathering-volatility-2-a-monthly-stress-test-to-guide-saving. The original *Weathering Volatility* (2015) was not opened.
15. The Pew Charitable Trusts. *Payday Lending in America: Who Borrows, Where They Borrow, and Why*, July 2012. https://www.pew.org/-/media/legacy/uploadedfiles/pcs_assets/2012/pewpaydaylendingreportpdf.pdf
16. Consumer Financial Protection Bureau. *Payday Loans and Deposit Advance Products: A White Paper of Initial Data Findings*, April 2013. https://files.consumerfinance.gov/f/201304_cfpb_payday-dap-whitepaper.pdf
17. CFPB. *CFPB Data Point: Payday Lending*, March 2014. https://files.consumerfinance.gov/f/201403_cfpb_report_payday-lending.pdf
18. 12 CFR §701.21(c)(7), Payday alternative loans (PALs I and II). https://www.ecfr.gov/current/title-12/chapter-VII/subchapter-A/part-701/section-701.21
19. NCUA. "NCUA Board Extends Loan Interest Rate Ceiling," Feb 6, 2026. https://ncua.gov/newsroom/press-release/2026/ncua-board-extends-loan-interest-rate-ceiling
20. U.S. Department of the Treasury. Daily Treasury Bill Rates, 2026 (row for 11/19/2026). https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2026
21. Mission Asset Fund. Lending Circles program page (accessed 2026-11-19). https://www.missionassetfund.org/lending-circles/
22. Abello, O. P. "'Genius' Money Idea Helps People Outside Big Banking's Inner Circle," *Next City*, Sept 29, 2016. https://nextcity.org/daily/entry/lending-circles-predatory-loans (secondary; source of the MAF 0.7% figure)
23. Reyes, B. & Lopez, E. "Social Lending: Improving Credit for the Underbanked," *Communities & Banking*, Federal Reserve Bank of Boston, Dec 2015. https://www.bostonfed.org/publications/communities-and-banking/2016/winter/social-lending-improving-credit-for-the-underbanked.aspx
24. Gugerty, M. K. "You Can't Save Alone: Commitment in Rotating Savings and Credit Associations in Kenya." *Economic Development and Cultural Change* 55(2): 251–282, 2007. https://doi.org/10.1086/508716. Full text is paywalled and was not read. The 6% exit figure comes from the summary in Basu (2008), below.
25. Basu, K. "Hyperbolic Discounting and the Sustainability of Rotational Savings Arrangements," 2008, MPRA Paper 20440. https://mpra.ub.uni-muenchen.de/20440/1/MPRA_paper_20440.pdf
26. Besley, T., Coate, S. & Loury, G. "The Economics of Rotating Savings and Credit Associations." *American Economic Review* 83(4): 792–810, 1993. https://www.jstor.org/stable/2117579 (theory; not opened)
27. McNabb, K., LeMay-Boucher, P. & Bonan, J. "Enforcement Problems in ROSCAs: Evidence from Benin." *European Journal of Development Research* 31: 1389–1415, 2019. https://ideas.repec.org/a/pal/eurjdr/v31y2019i5d10.1057_s41287-019-00215-5.html (abstract only)
28. Catholic Relief Services. *Group Performance in Fee-for-service Savings Groups* (SILC Innovations Research Brief 3), June 2012. https://www.crs.org/sites/default/files/2025-06/group-performance-in-fee-for-service-savings-groups.pdf
29. CARE. *VSLA by the Numbers*, 2022. https://www.care.org/wp-content/uploads/2022/11/VSLA-by-the-Numbers-Report_2022.pdf

### Not verified / could not load
- bls.gov HTML pages return 403 to scripted fetches. All BLS numbers come from the BLS Public Data API instead.
- No primary MAF document stating a default rate was found. The 0.7% figure is MAF's own claim as reported by the press.
- The Gugerty (2007) and Anderson, Baland & Moene (2009) full texts were paywalled or blocked, so no default rates from those papers are quoted directly.
- No US tanda study with quantitative default or dropout data was found.
- No source was found for `pepm_fee`, `opex_per_member_month`, `stake_prob`, `stake_cash_threshold` or `buffer_median`. They stay labeled as assumptions.
