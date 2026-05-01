"""
================================================================================
DATATHON 2026 - THE GRIDBREAKER
Enterprise-grade EDA & Visualization Pipeline for Fashion E-commerce
VinTelligence — VinUniversity DS&AI Club
================================================================================

v4 — STRATEGIST v3 FOLLOW-UP (2 new charts, 44–45 + bombshell reframing)
------------------------------------------------------------------------
  + Chart 44  Value-at-Stake quantified recommendations  (Prescriptive 7→10)
  + Chart 45  Probabilistic CLV + Churn distribution     (Predictive 6→8)
  + Chart 33 upgraded with shipping sensitivity panel    (CFO stress test)
  + Chart 00 upgraded with actual DQ results            (orphan FK + ship %0)
  + 🚨 BOMBSHELL banner at top: CM = −1.6% (or −10% to −12% at industry ship)
  + Section 2 Data Quality with 4 actual results tables
  + Section 5.1 competitive benchmarks (Industry · Inditex · Shein · ASOS)
  + Total: 46 charts

v3 — STRATEGIST FEEDBACK ADDRESSED (11 new charts, 33–43)
---------------------------------------------------------
  + Chart 33  Contribution Margin Waterfall   (Gap #1 — unit economics)
  + Chart 34  Revenue Decomposition 4-driver  (Gap #2 — Traffic×CVR×UPT×AUR)
  + Chart 35  COD Cancellation Diagnostic     (Gap #3 — VN context)
  + Chart 36  VN Calendar Event Lift          (Gap #3 — Tết, 11.11, 12.12…)

  + Chart 37  Full-Price STR @ Week 6         (Gap #5 — fashion #1 KPI)
  + Chart 38  Aged Inventory + Newness Ratio  (Gap #5 — Zara/Shein standard)
  + Chart 39  Category Penetration + 1st-LTV  (Gap #4 — behavior)
  + Chart 40  Wardrobing Flag + Order Split   (Gap #4 — fraud screen)
  + Chart 41  Price–COGS Paradox + Ladder     (Gap #5 — pricing root cause)
  + Chart 42  Review Keywords + NPS Proxy     (Gap #7 — text signal)
  + Chart 43  ESG: Carbon + Deadstock         (Gap #6 — sustainability)

v2 — research-driven baseline
-----------------------------
  Charts 00..32 (data quality, Pareto/Gini, KM survival, size curve,
  lifecycle S-curve, GMROI, promo DiD, STL anomaly).
  + INDUSTRY BENCHMARKS on every chart; PAIN-POINTS → SOLUTIONS matrix;
    METHODOLOGY REFERENCES (Fader, Cleveland, Taylor-Letham, etc.)

Storytelling Arc
----------------
    ACT 0   Data Quality & Integrity
    ACT 1   Business Pulse                  (Descriptive)
    ACT 2   Who Are Our Customers?          (Diagnostic)
    ACT 3   What Drives Revenue?            (Diagnostic -> Predictive)
    ACT 4   Where Are We Bleeding?          (Diagnostic)
    ACT 5   What Should We Do?              (Predictive -> Prescriptive)
    ACT 6   Industry-Grade Retail Metrics   (Benchmarked)
    ACT 7   Unit Economics                  (v3 — CFO layer)
    ACT 8   Vietnam Market Context          (v3 — localization)
    ACT 9   Fashion Merchandising           (v3 — velocity/newness/pricing)
    ACT 10  Customer Behavior Deep-Dive     (v3 — cross-sell/wardrobing)
    ACT 11  Text Mining & ESG               (v3 — reviews + sustainability)
    ACT 12  Decision Support                (v4 — Value-at-Stake + CLV)

Output
------
    figures/*.png     (52 static PNG charts, retina scale)
    EDA_REPORT.md     (consolidated markdown report with embedded charts)
    *.parquet         (analytical base tables for Part 3 forecasting)

Dependencies
------------
    pandas, numpy, plotly, kaleido, scipy, scikit-learn, statsmodels,
    networkx, pyarrow
================================================================================
"""

# =============================================================================
# 0. IMPORTS & CONFIG
# =============================================================================
import os
import warnings
from pathlib import Path
from datetime import timedelta
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"

# -----------------------------------------------------------------------------
# >>>  FILL THESE PATHS  <<<
# -----------------------------------------------------------------------------
DATA_DIR   = Path("/Users/leminh/Library/CloudStorage/GoogleDrive-lengoctuongvy4789@gmail.com/Drive của tôi/PERSONAL PROJECTS/VINUNI DATATHON/datathon-2026-round-1")
OUTPUT_DIR = Path("/Users/leminh/Library/CloudStorage/GoogleDrive-lengoctuongvy4789@gmail.com/Drive của tôi/PERSONAL PROJECTS/VINUNI DATATHON/OUTPUT3")
# -----------------------------------------------------------------------------

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PNG_WIDTH = 1280
PNG_SCALE = 2

PALETTE = {
    "primary": "#1F7A4C", "accent": "#E8A33D", "danger": "#C0392B",
    "info":    "#2E86C1", "muted":  "#7F8C8D", "bg":     "#FAFAFA",
    "seg": {
        "Champion":    "#1F7A4C", "Loyal": "#52BE80", "Potential": "#F5CBA7",
        "At-Risk":     "#E67E22", "Hibernating": "#85929E", "Lost": "#C0392B",
    },
}


# =============================================================================
# 1. INDUSTRY BENCHMARKS — fashion e-commerce norms (for chart annotations)
# =============================================================================
# Sources: McKinsey "State of Fashion" reports, BoF × McKinsey 2023-24,
#          Shopify Plus Fashion Report, Baymard Institute, Narvar returns data,
#          Nasdaq SpendEdge retail metrics, SSENSE / Zalando investor decks.
# -----------------------------------------------------------------------------
BENCHMARKS = {
    "return_rate_online_fashion":  0.25,   # 20-40% industry average
    "return_rate_healthy_target":  0.15,
    "gross_margin_fashion":        0.55,   # pure-play 50-65%
    "gross_margin_marketplace":    0.30,
    "repeat_rate_12m":             0.30,   # Shopify benchmark
    "cart_abandon_rate":           0.70,   # Baymard
    "checkout_conv_rate":          0.03,   # e-com median ~2-3%
    "full_price_sellthrough":      0.60,   # Zara ~85%, mass ~40-60%
    "gmroi_target":                1.5,    # retail rule-of-thumb
    "inventory_turns_fashion":     4.0,    # per year
    "aov_mass_fashion_usd":        50,
    "sessions_to_order_cvr":       0.025,
    "size_curve_norm": {"S": 15, "M": 35, "L": 35, "XL": 15},  # industry avg mix %
    "product_lifecycle_weeks_fast_fashion":   3,   # Shein / Zara
    "product_lifecycle_weeks_traditional":   12,
    "ltv_cac_ratio_healthy":       3.0,
    # --- v3 additions — variable cost stack (Reichheld & Schefter HBR 2000) ---
    "shipping_cost_pct":           0.10,   # 8-12% of revenue
    "payment_processing_pct":      0.025,  # 1.5-3% (weighted avg)
    "packaging_cost_pct":          0.04,   # 3-5%
    "return_processing_pct":       0.018,  # 1.5-2% of revenue
    # --- v3 additions — Vietnam / SEA specifics ---
    "cod_cancel_rate_vn":          0.25,   # 20-30% (Ninja Van 2023)
    "cod_share_vn":                0.50,   # 40-60% VN fashion is COD
    # --- v3 additions — Fashion merchandising ---
    "full_price_str_week6_healthy":0.60,   # Fisher & Raman HBR 1994
    "newness_ratio_zara_shein":    0.75,   # 70-80% revenue <90-days-old
    "aged_inventory_warn_days":    90,
    "deadstock_days":              180,
    "wardrobing_return_rate":      0.50,   # individual customer threshold
    # --- v3 additions — ESG (UK Environment Agency / BoF) ---
    "carbon_per_return_kg":        0.5,
    # --- Payment processing fees by method (industry averages) ---
    "payment_fee_rates": {
        "credit_card":   0.025,
        "paypal":        0.030,
        "bank_transfer": 0.010,
        "cod":           0.005,   # not a processing fee but logistic handling
    },
}

# -----------------------------------------------------------------------------
# Vietnam calendar — hard-coded major event windows (v3 Gap #3 fix)
#   Tết dates change lunar-to-Gregorian; below are verified dates 2013-2024.
# -----------------------------------------------------------------------------
TET_DATES = {
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
}
VN_FIXED_EVENTS = [
    ("Valentine",   "02-14", 1),
    ("Int'l Women's Day (8/3)",  "03-08", 1),
    ("Reunification+Labour (30/4-1/5)", "04-30", 3),
    ("VN National Day (2/9)", "09-02", 1),
    ("VN Women's Day (20/10)","10-20", 1),
    ("11.11 (Shopping Day)",  "11-11", 1),
    ("Black Friday",          "11-24", 4),   # approx last Fri Nov
    ("12.12 (Shopping Day)",  "12-12", 1),
    ("Christmas",             "12-25", 3),
]


# =============================================================================
# 2. FASHION E-COMMERCE PAIN POINTS → SOLUTIONS
# =============================================================================
# Mapping: which operational pain each chart addresses.
# Each pain tagged with the canonical research / playbook that inspired it.
# -----------------------------------------------------------------------------
PAIN_POINTS = {
    "P1_returns": dict(
        name  ="Returns & reverse logistics cost",
        fact  ="Online fashion return rate 20-40% (Narvar 2023); reverse logistics "
               "costs 15-25% of revenue.",
        lever ="Size/fit algorithm (ASOS Fit Assistant), quality control at SKU level, "
               "return-reason root-cause, customer return profiling.",
    ),
    "P2_inventory": dict(
        name  ="Inventory obsolescence & markdown erosion",
        fact  ="Fashion SKU shelf life 8-12 weeks; unsold inventory triggers 40-60% "
               "markdown (Inditex avg markdown 15% vs industry 35%).",
        lever ="Weeks-of-supply tracking, sell-through velocity curves, GMROI per "
               "category, automated markdown triggers.",
    ),
    "P3_cold_start": dict(
        name  ="New-product cold-start & trend lifecycle",
        fact  ="Shein introduces 6,000 SKUs/day; traditional brands struggle to "
               "identify rising SKUs within the 2-week decision window.",
        lever ="Launch-normalized S-curves, early-signal attention on week-1 "
               "velocity, trend decay detection.",
    ),
    "P4_fit_size": dict(
        name  ="Fit & size distribution mismatch",
        fact  ="Size/fit is #1 return reason (~70% of returns, Zalando 2022 report).",
        lever ="Size curve optimization vs demand signal, product-level fit scores "
               "from review rating × return rate.",
    ),
    "P5_customer_churn": dict(
        name  ="Customer churn & retention decay",
        fact  ="Fashion repeat rate 20-30% industry avg (Shopify). Top 10% of "
               "customers drive 40-60% of revenue (Pareto).",
        lever ="RFM + CLV segmentation, Kaplan-Meier survival, cohort retention, "
               "churn x CLV action matrix.",
    ),
    "P6_promo_efficacy": dict(
        name  ="Promotion efficacy & cannibalization",
        fact  ="30-50% of promo revenue is merely pull-forward of baseline demand "
               "(HBR, Blattberg & Neslin 1990).",
        lever ="Incrementality testing (DiD), promo ROI per campaign, elasticity "
               "curves, kill-list for bottom-quintile promos.",
    ),
    "P7_channel_attribution": dict(
        name  ="Multi-touch channel attribution",
        fact  ="Last-click attribution over-credits paid search by 30-50% "
               "(Google 2021 MMM study).",
        lever ="Acquisition-channel CLV, cohort-level LTV:CAC, leading-indicator "
               "lag correlation with traffic.",
    ),
    "P8_geographic_leak": dict(
        name  ="Shipping SLA leakage & geographic coverage",
        fact  ="Late delivery correlates r≈0.4 with cancellation/return "
               "(MIT Sloan logistics research).",
        lever ="Lead-time distributions per region, shipping-cancel regression.",
    ),
    "P9_data_quality": dict(
        name  ="Silent data quality decay",
        fact  ="80% of data-science failures trace to data-quality issues at the "
               "join layer (Google Data-ML Tech Debt paper, Sculley et al. 2015).",
        lever ="Up-front missing-rate audit, referential-integrity checks, date "
               "coverage tracking.",
    ),
    "P10_concentration_risk": dict(
        name  ="Revenue concentration risk",
        fact  ="If top-5% of SKUs or customers disappear, business fragile. "
               "Gini > 0.8 = extreme concentration.",
        lever ="SKU Pareto + Gini, customer Lorenz curve, HHI per category.",
    ),
    # --- v3 additions — strategist feedback gaps ---
    "P11_unit_economics": dict(
        name  ="Hidden variable costs eat gross profit",
        fact  ="Reichheld & Schefter (HBR 2000): gross margin 30% business can "
               "burn ALL of it on shipping + returns. Contribution margin, "
               "not gross margin, is the truth layer.",
        lever ="Contribution-margin waterfall: -shipping -payment fees "
               "-packaging -return processing. CFO's #1 question.",
    ),
    "P12_revenue_driver_blindness": dict(
        name  ="Revenue = Traffic × CVR × UPT × AUR (4-driver decomposition)",
        fact  ="Levy & Weitz Retailing Management Ch.5. When revenue dips, the "
               "RIGHT lever depends on WHICH driver slipped.",
        lever ="Monthly 4-driver index chart + deviation attribution.",
    ),
    "P13_cod_cancellation_vn": dict(
        name  ="COD cancellation — unique VN risk",
        fact  ="40-60% VN fashion orders are COD; 20-30% of COD gets cancelled "
               "(Ninja Van SEA Logistics 2023). Burn logistics + lost margin.",
        lever ="Cancel-rate diagnostic by payment × region × AOV bucket; "
               "COD-deposit experiments; pre-delivery cancel flag.",
    ),
    "P14_vn_calendar": dict(
        name  ="Vietnam-specific calendar events drive disproportionate demand",
        fact  ="Tết, 8/3, 20/10, 11.11, 12.12 can lift revenue 2-5× baseline. "
               "Inventory planning blind to these = stockout or deadstock.",
        lever ="Calendar-event lift measurement; Prophet regressors; "
               "buying plan calibrated to VN holiday calendar.",
    ),
    "P15_fashion_velocity": dict(
        name  ="Sell-through velocity & newness — fashion's #1 KPI",
        fact  ="Fisher & Raman (HBR 1994): Week-6 full-price sell-through "
               "predicts full-season margin. Zara/Shein: 70-80% revenue from "
               "products <90 days old (newness ratio).",
        lever ="Full-price STR at Week 6 per SKU, aged-inventory buckets "
               "(0-30, 30-60, 60-90, 90-180, 180+), newness ratio over time.",
    ),
    "P16_wardrobing_fraud": dict(
        name  ="Wardrobing / serial returners",
        fact  ="Appriss Retail 2023: 6-12% of returns are fraud/wardrobing "
               "(wear once → return). Invisible in aggregate return rate.",
        lever ="Per-customer return rate; flag top 1% with return-rate >50% "
               "and >=3 orders; dynamic return policy.",
    ),
    "P17_esg_risk": dict(
        name  ="ESG / sustainability material risk",
        fact  ="McKinsey State of Fashion 2024: sustainability is financial "
               "material. 73% Gen Z pays premium for sustainable brands "
               "(IBM 2022). Shein under EU investigation for greenwashing.",
        lever ="Return carbon footprint proxy, deadstock risk tracking, "
               "aged-inventory write-down forecast.",
    ),
    "P18_text_signal": dict(
        name  ="Ignored text signal in reviews",
        fact  ="Ghose & Ipeirotis (MIS Quarterly 2011): review TEXT predicts "
               "returns/churn stronger than rating alone.",
        lever ="Keyword extraction by rating bucket, NPS proxy, theme "
               "clustering for product-quality feedback.",
    ),
}


# =============================================================================
# 3. METHODOLOGY REFERENCES — the 'academic/industry spine' of each chart
# =============================================================================
METHODOLOGY_REFS = [
    ("RFM Segmentation",
     "Bult & Wansbeek (1995), Hughes (1994). Used by Amazon, Sephora, Nike."),
    ("Probabilistic CLV (BG/NBD + Gamma-Gamma)",
     "Fader & Hardie (2005) 'Counting Your Customers the Easy Way'. "
     "Implemented in Python `lifetimes` package."),
    ("Cohort Retention Heatmap",
     "Andrew Chen (Andreessen Horowitz) retention framework; "
     "standard in Amplitude, Mixpanel."),
    ("STL Decomposition",
     "Cleveland, Cleveland, McRae & Terpenning (1990) 'STL: A Seasonal-Trend "
     "Decomposition Procedure Based on Loess'."),
    ("Prophet / Forecasting at Scale",
     "Taylor & Letham (2017) Meta AI. Handles holidays, change-points, "
     "trend saturation."),
    ("Kaplan-Meier Survival",
     "Kaplan & Meier (1958). Applied to customer churn by Fader (HBS)."),
    ("Price Elasticity",
     "Deaton (1988) demand estimation; Nevo (2001) retail elasticity."),
    ("Promotion Incrementality (DiD)",
     "Blattberg & Neslin (1990) 'Sales Promotion'; Lewis & Reiley (2014) "
     "on-line ads causal measurement."),
    ("BCG Growth-Share Matrix",
     "Henderson (1970) Boston Consulting Group; still used in CPG "
     "portfolio reviews."),
    ("Market Basket Analysis",
     "Agrawal & Srikant (1994) Apriori; FP-Growth (Han et al. 2000)."),
    ("GMROI & Retail KPIs",
     "Levy & Weitz (2012) 'Retailing Management', McKinsey Retail Practice."),
    ("Pareto / Gini Concentration",
     "Lorenz (1905); Gini (1912). Standard in actuarial & retail risk."),
    ("Size Curve Optimization",
     "Zalando Research (2019), ASOS Fit Assistant tech papers."),
    ("Product Lifecycle S-Curves",
     "Rogers (1962) Diffusion of Innovations; Fisher & Raman (2010) "
     "'The New Science of Retailing'."),
    ("Anomaly Detection via STL Residual",
     "Hochenbaum et al. (2017) Twitter's SH-ESD; Cleveland (1990) STL."),
    ("Customer Concentration Risk (HHI)",
     "Herfindahl-Hirschman Index — DOJ/FTC antitrust, applied to revenue."),
    # ---- v3 additions ----
    ("Contribution Margin Waterfall",
     "Reichheld & Schefter (2000) 'E-Loyalty: Your Secret Weapon on the Web' "
     "Harvard Business Review."),
    ("Revenue 4-driver decomposition",
     "Levy & Weitz (2012) 'Retailing Management' Ch.5; Shopify Commerce "
     "Benchmarks."),
    ("Full-Price Sell-Through @ Week 6",
     "Fisher & Raman (1994) 'Making Supply Meet Demand in an Uncertain World' "
     "HBR; 'The New Science of Retailing' (2010)."),
    ("Customer LTV by First-Purchase Category",
     "Kumar & Shah (2004) 'Building and Sustaining Profitable Customer Loyalty "
     "for the 21st Century'."),
    ("NPS Proxy",
     "Reichheld (2003) 'The One Number You Need to Grow' HBR."),
    ("Wardrobing / Serial Returner detection",
     "Appriss Retail (2023) 'Consumer Returns in the Retail Industry'."),
    ("VN COD Cancellation Benchmark",
     "Ninja Van SEA Logistics Report (2023); Momentum Works (2024) on VN "
     "live-commerce."),
    ("Return Carbon Footprint",
     "UK Environment Agency Life-Cycle Assessment guidelines; BoF "
     "Sustainability Index."),
    # ---- v4 additions ----
    ("Value-at-Stake Quantified Recommendations",
     "McKinsey 'Value-at-Stake' methodology; RICE prioritization (Intercom)."),
    ("Growth-Trap Diagnosis",
     "Christensen (1997) 'The Innovator's Dilemma'; case studies WeWork / "
     "Casper / Blue Apron / Wish."),
    ("Sensitivity Analysis on Variable Costs",
     "Standard CFO practice for unit-economics stress-testing under data "
     "ambiguity."),
]


# Best-in-class competitive benchmarks (v4) — for 3-way comparison
COMPETITIVE_BM = {
    "gross_margin": dict(industry=0.55, inditex=0.60, shein=0.40, asos=0.45),
    "return_rate":  dict(industry=0.25, inditex=0.10, shein=0.15, asos=0.30),
    "newness_ratio":dict(industry=0.40, inditex=0.65, shein=0.85, asos=0.50),
    "full_price_str":dict(industry=0.55, inditex=0.85, shein=0.70, asos=0.50),
}


# =============================================================================
# 4. CHART METADATA — drives rendering & the MD report
# =============================================================================
CHART_META = {
    # ---------------- ACT 0 — Data Integrity ----------------
    "00_data_quality": dict(
        title="Data Quality Audit",
        act  ="ACT 0 — Data Quality & Integrity",
        sources=["(all 13 tables)"],
        meaning="Heatmap null-rate + coverage window + referential-integrity check. "
                "Gate: nothing below passes without this first.",
        pain="P9_data_quality",
        benchmark="Null-rate <5% for mandatory fields; referential integrity 100%.",
        reference="Sculley et al. 2015 'Hidden Technical Debt in ML Systems'."),

    # ---------------- ACT 1 — Business Pulse ----------------
    "01_kpi_scorecard": dict(
        title="Executive KPI Scorecard",
        act  ="ACT 1 — Business Pulse",
        sources=["sales","orders","order_items","products","returns","customers"],
        meaning="Sáu chỉ số lifetime đóng vai trò 'health-check' toàn doanh nghiệp.",
        pain="—",
        benchmark=f"Gross Margin fashion {BENCHMARKS['gross_margin_fashion']:.0%}; "
                  f"Repeat 12m {BENCHMARKS['repeat_rate_12m']:.0%}.",
        reference="McKinsey State of Fashion 2024."),
    "02_revenue_waterfall": dict(
        title="Revenue Waterfall",
        act  ="ACT 1 — Business Pulse",
        sources=["order_items","products","promotions","returns"],
        meaning="Đi từ Gross Revenue xuống Gross Profit qua 4 bước, định lượng rò rỉ.",
        pain="P1_returns",
        benchmark="Refund/Revenue <3% healthy; Promo depth <10% for full-price brands.",
        reference="Retail P&L waterfall — Levy & Weitz (2012)."),
    "03_revenue_stl": dict(
        title="Revenue Timeline — STL + promo overlays",
        act  ="ACT 1 — Business Pulse",
        sources=["sales","promotions"],
        meaning="Tách trend / seasonality / residual; overlay promo lớn để đối chiếu lift.",
        pain="—",
        benchmark="STL period=365 cho retail daily; residual σ là baseline cho anomaly.",
        reference="Cleveland et al. 1990 STL paper."),
    "04_yoy_heatmap": dict(
        title="Year × Month YoY Growth",
        act  ="ACT 1 — Business Pulse",
        sources=["sales"],
        meaning="Tháng nào chậm lại theo năm — tín hiệu sớm của saturation / leak.",
        pain="—",
        benchmark="E-com fashion Vietnam ~15-25% YoY 2020-2024 (Euromonitor).",
        reference="Standard YoY heatmap — Tableau / Looker retail templates."),
    "05_dow_month": dict(
        title="Day-of-Week × Month seasonality",
        act  ="ACT 1 — Business Pulse",
        sources=["sales"],
        meaning="Mùa vụ intra-week → lịch ad-spend, staff, replenishment.",
        pain="—",
        benchmark="Fashion e-com peak thứ 5-CN; dip thứ 2-3.",
        reference="Shopify Plus weekly cycle report."),

    # ---------------- ACT 2 — Customers ----------------
    "06_rfm_scatter": dict(
        title="RFM Segmentation — 6-segment map",
        act  ="ACT 2 — Who Are Our Customers?",
        sources=["customers","orders","order_items","products"],
        meaning="Phân khúc 6 nhóm (Champion/Loyal/Potential/At-Risk/Hibernating/Lost).",
        pain="P5_customer_churn",
        benchmark="Champions+Loyal thường 20-30% base; nếu <10% → retention yếu.",
        reference="Hughes 1994, Bult & Wansbeek 1995."),
    "07_cohort_retention": dict(
        title="Cohort Retention Heatmap",
        act  ="ACT 2 — Who Are Our Customers?",
        sources=["customers","orders"],
        meaning="Retention diagonal — đo tốc độ churn & CAC payback.",
        pain="P5_customer_churn",
        benchmark="M3 retention >25% healthy; M12 >12% strong (Amplitude 2023).",
        reference="Andrew Chen retention framework."),
    "08_clv_channel": dict(
        title="CLV by Acquisition Channel",
        act  ="ACT 2 — Who Are Our Customers?",
        sources=["customers","orders","order_items"],
        meaning="Kênh nào đem khách giá trị cao nhất — không chỉ nhiều nhất.",
        pain="P7_channel_attribution",
        benchmark="LTV:CAC ≥3:1 healthy; <1 burn-mode.",
        reference="David Skok (SaaS KPIs) adapted to retail."),
    "09_demographic_sunburst": dict(
        title="Demographic × Product Revenue Sunburst",
        act  ="ACT 2 — Who Are Our Customers?",
        sources=["customers","orders","order_items","products"],
        meaning="Gender → Age → Category → Segment: phát hiện pocket revenue tập trung.",
        pain="—",
        benchmark="—",
        reference="Hyper-segmentation playbook — Stitch Fix."),
    "10_geo_map": dict(
        title="Geographic Revenue Distribution",
        act  ="ACT 2 — Who Are Our Customers?",
        sources=["orders","order_items","geography","products","customers"],
        meaning="Treemap region/city + top-15 Revenue/Customer — bản đồ ưu tiên mở rộng.",
        pain="P8_geographic_leak",
        benchmark="Top-3 cities thường chiếm 50-70% revenue fashion e-com VN.",
        reference="Euromonitor 2024 Vietnam apparel report."),

    # ---------------- ACT 3 — Revenue Drivers ----------------
    "11_bcg_matrix": dict(
        title="BCG Growth-Share Matrix — category portfolio",
        act  ="ACT 3 — What Drives Revenue?",
        sources=["order_items","products"],
        meaning="Stars / Cash Cows / Questions / Dogs — budget allocation.",
        pain="P2_inventory",
        benchmark="Stars target: share >median & YoY >15%.",
        reference="Henderson 1970 BCG."),
    "12_price_elasticity": dict(
        title="Price Elasticity — units by discount depth",
        act  ="ACT 3 — What Drives Revenue?",
        sources=["order_items","promotions","products"],
        meaning="Đường cong units theo discount bin — sweet-spot tối đa hoá revenue.",
        pain="P6_promo_efficacy",
        benchmark="Elasticity apparel ≈ −1.5 to −2.5 (Nevo 2001).",
        reference="Deaton 1988, Nevo 2001."),
    "13_promo_roi": dict(
        title="Promotion ROI Landscape",
        act  ="ACT 3 — What Drives Revenue?",
        sources=["promotions","order_items","orders"],
        meaning="Discount-cost vs revenue, màu = ROI → cut bottom quintile.",
        pain="P6_promo_efficacy",
        benchmark="ROAS promo ≥4 healthy (Meta/Google e-com ads).",
        reference="Blattberg & Neslin 1990.",
        caveat="Per `[A14_discount_math]` audit: 5/50 promos là `fixed` type "
               "(`discount_value` as VND, không phải %). Aggregate ROI valid vì dùng "
               "`discount_amount` field (always accurate downstream). Per-promo-type "
               "breakdown excludes fixed promos do unit ambiguity — pending BTC "
               "clarification on whether `discount_value` cho fixed type là scaled "
               "VND hay raw VND. 36% reconciliation gap (45/126 promo lines unfit "
               "formula) cũng documented — chart aggregate-level vẫn valid."),
    "14_return_heatmap": dict(
        title="Return-rate heatmap — Category × Size",
        act  ="ACT 3 — What Drives Revenue?",
        sources=["products","order_items","returns"],
        meaning="Ô (category × size) có tỉ lệ trả hàng cao = vấn đề fit/quality.",
        pain="P4_fit_size",
        benchmark=f"Industry return rate {BENCHMARKS['return_rate_online_fashion']:.0%}; "
                  f"Healthy target {BENCHMARKS['return_rate_healthy_target']:.0%}.",
        reference="Zalando 2022 return root-cause analysis."),
    "15_copurchase_network": dict(
        title="Co-purchase Network — Category affinity",
        act  ="ACT 3 — What Drives Revenue?",
        sources=["order_items","products"],
        meaning="Đồ thị đồng-mua → input cho cross-sell, bundle, PDP recommendation.",
        pain="—",
        benchmark="Lift ≥2 = actionable affinity (Apriori).",
        reference="Agrawal & Srikant 1994."),

    # ---------------- ACT 4 — Operational Leaks ----------------
    "16_funnel": dict(
        title="Traffic → Order Funnel",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["web_traffic","orders","order_items"],
        meaning="Sessions → Orders tổng + CVR theo source.",
        pain="P7_channel_attribution",
        benchmark=f"E-com median CVR {BENCHMARKS['sessions_to_order_cvr']:.1%}; "
                  f"Cart-abandon {BENCHMARKS['cart_abandon_rate']:.0%}.",
        reference="Baymard Institute 2023 benchmarks."),
    "17_inventory_quadrant": dict(
        title="Inventory Health Quadrant",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["inventory","order_items","products"],
        meaning="DoS × sell-through, bubble = lost-sales ước tính.",
        pain="P2_inventory",
        benchmark="DoS 30-60 healthy; >120 → markdown risk.",
        reference="Fisher & Raman 2010 'New Science of Retailing'."),
    "18_dos_sellthrough": dict(
        title="Days-of-Supply vs Sell-through density",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["inventory"],
        meaning="Phát hiện cluster SKU overstock — plan markdown.",
        pain="P2_inventory",
        benchmark=f"Healthy sell-through tuần >{BENCHMARKS['full_price_sellthrough']:.0%}.",
        reference="Inditex operational model."),
    "19_returns_sankey": dict(
        title="Return flow — Category → Reason → Refund size",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["returns","order_items","products"],
        meaning="Refund chảy từ đâu, qua lý do nào, đến size nào — root-cause.",
        pain="P1_returns",
        benchmark="Size/fit ~70% return reasons (Narvar 2023).",
        reference="Narvar 'State of Returns 2023'."),
    "20_shipping_sla": dict(
        title="Shipping SLA — lead time by region",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["shipments","orders","geography"],
        meaning="Vùng vượt SLA → tăng return/cancel.",
        pain="P8_geographic_leak",
        benchmark="HCMC/HN target ≤2 ngày; tỉnh xa ≤5.",
        reference="MIT Sloan logistics 2020."),
    "21_review_return": dict(
        title="Quality signal — Rating vs Return rate",
        act  ="ACT 4 — Where Are We Bleeding?",
        sources=["reviews","returns","order_items","products"],
        meaning="Scatter product-level với OLS trendline — kiểm định giả thuyết.",
        pain="P4_fit_size",
        benchmark="r < −0.3 = rating là predictor mạnh cho return.",
        reference="Ghose & Ipeirotis 2011 review mining."),

    # ---------------- ACT 5 — Strategic Outlook ----------------
    "22_forecast_preview": dict(
        title="Forecast Preview — seasonal-naïve baseline",
        act  ="ACT 5 — What Should We Do?",
        sources=["sales"],
        meaning="Baseline + 95% band — đối chứng cho Part 3.",
        pain="—",
        benchmark="Seasonal-naïve MAPE ≈ 20-25% retail baseline.",
        reference="Hyndman & Athanasopoulos 'Forecasting: Principles & Practice'."),
    "23_churn_clv": dict(
        title="Churn × CLV Action Matrix",
        act  ="ACT 5 — What Should We Do?",
        sources=["customers","orders","order_items"],
        meaning="4-ô hành động: RETAIN / NURTURE / GROW / LET-GO.",
        pain="P5_customer_churn",
        benchmark="Retain quadrant thường 5-10% base — concentrate spend.",
        reference="Kumar & Shah 2004 CLV management."),
    "24_traffic_lag": dict(
        title="Traffic → Revenue lead/lag CCF",
        act  ="ACT 5 — What Should We Do?",
        sources=["web_traffic","sales"],
        meaning="Cross-correlation → leading indicator thời gian.",
        pain="—",
        benchmark="Peak lag 0-3 ngày điển hình e-com.",
        reference="Box-Jenkins CCF methodology."),
    "25_action_priority": dict(
        title="Action Priority Matrix — Impact × Effort",
        act  ="ACT 5 — What Should We Do?",
        sources=["(meta-analysis)"],
        meaning="Impact × Effort của đề xuất prescriptive.",
        pain="—",
        benchmark="—",
        reference="RICE framework (Intercom)."),

    # ---------------- ACT 6 — Retail-grade Metrics (NEW) ----------------
    "26_sku_pareto": dict(
        title="SKU Pareto & Gini — revenue concentration",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["order_items","products"],
        meaning="Lorenz curve revenue-cumulative; Gini đo inequality. "
                "Phát hiện 80/20 thực tế và rủi ro concentration.",
        pain="P10_concentration_risk",
        benchmark="Retail Gini 0.6-0.75 bình thường; >0.85 = phụ thuộc long-tail nguy hiểm.",
        reference="Lorenz 1905, Gini 1912; Anderson 'Long Tail' 2006."),
    "27_survival_curves": dict(
        title="Kaplan-Meier Survival — activation & 2nd-order",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["customers","orders"],
        meaning="Tỉ lệ khách 'sống' (chưa mua) vs thời gian từ signup / đơn đầu tiên. "
                "Phân tách theo kênh — kênh nào activate nhanh nhất.",
        pain="P5_customer_churn",
        benchmark="Median activation <30 ngày = kênh chất lượng; >90 ngày = cần nurture.",
        reference="Kaplan & Meier 1958; Fader HBS customer-base analysis."),
    "28_size_curve": dict(
        title="Size Curve Deviation — actual vs industry norm",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["order_items","products"],
        meaning="% bán theo size (S/M/L/XL) mỗi category so với norm ngành "
                "(S=15, M=35, L=35, XL=15). Gap = buying plan cần điều chỉnh.",
        pain="P4_fit_size",
        benchmark="Deviation >10pp vs norm ở 1 size → mismatch cung-cầu.",
        reference="Zalando Research 2019 size curve; ASOS Fit Assistant."),
    "29_lifecycle_scurve": dict(
        title="Product Lifecycle — launch-normalized S-curves",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["order_items","products"],
        meaning="Chuẩn hoá doanh số SKU từ tuần 0 (ra mắt). Phát hiện tốc độ "
                "peak & decay — nhận diện trend-cycle per category.",
        pain="P3_cold_start",
        benchmark=f"Fast-fashion peak ≈ tuần {BENCHMARKS['product_lifecycle_weeks_fast_fashion']}; "
                  f"truyền thống ≈ tuần {BENCHMARKS['product_lifecycle_weeks_traditional']}.",
        reference="Fisher & Raman 2010; Shein trend-cycle studies."),
    "30_gmroi": dict(
        title="GMROI per Category — gross margin on inventory investment",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["order_items","products","inventory"],
        meaning="Gross Profit / Avg Inventory-at-cost. Là chuẩn vàng retail — "
                "category nào sinh lời nhất trên mỗi đồng vốn tồn kho.",
        pain="P2_inventory",
        benchmark=f"GMROI target ≥{BENCHMARKS['gmroi_target']}; "
                  f"<1.0 = mất vốn vận hành.",
        reference="Levy & Weitz 'Retailing Management' chương 9."),
    "31_promo_incrementality": dict(
        title="Promotion Incrementality — DiD proxy",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["order_items","promotions"],
        meaning="Doanh thu trung bình ngày-promo vs ngày-không-promo cùng SKU, "
                "tổng hợp per category → % lift 'thực' sau khi trừ baseline.",
        pain="P6_promo_efficacy",
        benchmark="Incremental lift <30% = cannibalization chiếm đa số.",
        reference="Blattberg & Neslin 1990; Lewis & Reiley 2014."),
    "32_anomaly_stl": dict(
        title="Anomaly Detection — STL residual z-score",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["sales","promotions","web_traffic"],
        meaning="Residual |z|>3 trên STL → ngày bất thường có thể quy cho promo, "
                "traffic spike, hoặc event vĩ mô.",
        pain="—",
        benchmark="Healthy: <2% ngày là anomaly; >5% = model dưới-phủ.",
        reference="Cleveland 1990; Hochenbaum Twitter S-H-ESD 2017."),

    # ---------------- ACT 7 — Unit Economics (v3 strategist feedback) -----------
    "33_contribution_margin": dict(
        title="Contribution Margin Waterfall — the truth layer under Gross Profit",
        act  ="ACT 7 — Unit Economics (v3)",
        sources=["order_items","products","shipments","payments","returns"],
        meaning="Đi từ Net Revenue → −COGS → −Shipping → −Payment fees → "
                "−Packaging → −Return processing → Contribution Margin. "
                "CFO layer — con số lãi THẬT sau chi phí biến đổi.",
        pain="P11_unit_economics",
        benchmark="Shipping 8-12%, Payment 1.5-3%, Packaging 3-5%, Returns 1.5-2% "
                  "(Reichheld & Schefter HBR 2000).",
        reference="Reichheld & Schefter 'E-Loyalty' HBR 2000."),
    "34_revenue_decomposition": dict(
        title="Revenue Decomposition — Traffic × CVR × UPT × AUR (4 drivers)",
        act  ="ACT 7 — Unit Economics (v3)",
        sources=["web_traffic","orders","order_items"],
        meaning="Chỉ số index hoá từ tháng đầu = 100. Khi revenue dip, driver nào "
                "đang kéo xuống (Traffic marketing, CVR UX, UPT cross-sell, AUR "
                "pricing) quyết định đúng hành động.",
        pain="P12_revenue_driver_blindness",
        benchmark="Standard retail decomposition — Levy & Weitz Ch.5.",
        reference="Shopify Commerce Benchmarks; Levy & Weitz Retailing."),

    # ---------------- ACT 8 — Vietnam Market Context (v3) ----------------
    "35_cod_cancellation": dict(
        title="COD Cancellation Diagnostic — payment × region × AOV",
        act  ="ACT 8 — Vietnam Market Context (v3)",
        sources=["orders","payments","geography","order_items"],
        meaning="Cancel-rate per payment_method × region — phát hiện 'vùng-lỗ' "
                "nơi đốt logistics cost cho đơn không nhận.",
        pain="P13_cod_cancellation_vn",
        benchmark="COD cancel rate VN 20-30% (Ninja Van 2023). "
                  "Non-COD typically <5%.",
        reference="Ninja Van SEA Logistics Report 2023; Momentum Works 2024."),
    "36_vn_calendar_lift": dict(
        title="Vietnam Calendar Event Lift — Tết, 11.11, 12.12, 8/3, 20/10",
        act  ="ACT 8 — Vietnam Market Context (v3)",
        sources=["sales","(VN calendar hard-coded)"],
        meaning="% lift doanh thu trong cửa sổ event vs baseline ±30d. Event nào "
                "thật sự đáng đầu tư inventory/marketing.",
        pain="P14_vn_calendar",
        benchmark="Tết, 11.11 thường lift 2-5×; 8/3, 20/10 1.2-2× (VN e-com).",
        reference="Momentum Works 2024; local e-com platform reports."),

    # ---------------- ACT 9 — Fashion Merchandising (v3) ----------------
    "37_str_week6": dict(
        title="Full-Price Sell-Through at Week 6 — ngành #1 KPI",
        act  ="ACT 9 — Fashion Merchandising (v3)",
        sources=["order_items","inventory","products"],
        meaning="% units bán full-price trong 42 ngày đầu / tổng units_received. "
                "KPI then-chốt của Zara/Inditex; threshold 60% = healthy buy.",
        pain="P15_fashion_velocity",
        benchmark=f"Fisher & Raman (HBR 1994) "
                  f"threshold {BENCHMARKS['full_price_str_week6_healthy']:.0%}.",
        reference="Fisher & Raman 'Accurate Response' HBR 1994 + "
                  "'The New Science of Retailing' 2010."),
    "38_aged_newness": dict(
        title="Aged Inventory Buckets + Newness Ratio",
        act  ="ACT 9 — Fashion Merchandising (v3)",
        sources=["inventory","order_items","products"],
        meaning="Giá trị tồn kho theo tuổi (0-30, 30-60, 60-90, 90-180, 180+) "
                "+ % revenue từ sản phẩm <90 ngày theo quý. Đo 'tươi mới' của "
                "catalogue.",
        pain="P15_fashion_velocity",
        benchmark=f"Newness target Zara/Shein "
                  f"{BENCHMARKS['newness_ratio_zara_shein']:.0%}; "
                  f"aged >{BENCHMARKS['deadstock_days']}d = deadstock.",
        reference="Inditex investor decks; Shein 7-day cycle analyses."),
    "41_price_paradox": dict(
        title="Price–COGS Paradox — 3-layer margin compression",
        act  ="ACT 9 — Fashion Merchandising (v3)",
        sources=["products","order_items"],
        meaning="Margin compression: full-catalog list 30.6% → sold-only list 19.8% "
                "(−10.8pp selection bias from 814 unsold high-margin catalog padding) "
                "→ realized 14.6% (−5.2pp discount erosion). Two distinct losses, "
                "different remediation strategies — selection bias is data scope; "
                "discount erosion is pricing/promo policy.",
        pain="P11_unit_economics",
        benchmark="Healthy cogs/price <0.55; >0.80 = cần đàm phán lại supplier.",
        reference="Good-Better-Best price architecture; Simon-Kucher pricing."),

    # ---------------- ACT 10 — Customer Behavior Deep-Dive (v3) ----------------
    "39_category_penetration": dict(
        title="Category Penetration + First-Purchase → LTV",
        act  ="ACT 10 — Customer Behavior Deep-Dive (v3)",
        sources=["customers","orders","order_items","products"],
        meaning="Ma trận P(mua category B | đã mua A) — input cho cross-sell. "
                "Kèm median LTV theo category đầu tiên — predictor mạnh.",
        pain="—",
        benchmark="Khách mua ≥2 category có CLV 3-4× khách 1-category "
                  "(Kumar & Shah 2004).",
        reference="Kumar & Shah 2004 'Customer Lifetime Value Management'."),
    "40_wardrobing": dict(
        title="Wardrobing Flag + Cancel vs Return split",
        act  ="ACT 10 — Customer Behavior Deep-Dive (v3)",
        sources=["orders","returns","customers","order_items"],
        meaning="Phân phối return-rate per customer; flag wardrobers (>50% RR, "
                "≥3 orders). Cùng panel: pre-delivery cancel vs post-delivery "
                "return split — giải thích paradox return rate 5.59% thấp bất thường.",
        pain="P16_wardrobing_fraud",
        benchmark="Appriss Retail 2023: wardrobing 6-12% of returns volume.",
        reference="Appriss Retail 'Consumer Returns in the Retail Industry 2023'."),

    # ---------------- ACT 11 — Review Text & ESG (v3) ----------------
    "42_review_nps": dict(
        title="Review Keywords by Rating + NPS Proxy",
        act  ="ACT 11 — Text Mining & ESG (v3)",
        sources=["reviews"],
        meaning="Top keyword per rating bucket (Negative 1-2★, Neutral 3★, "
                "Positive 4-5★) + NPS proxy = (%5★) − (%1-3★). Text signal "
                "thường mạnh hơn rating số.",
        pain="P18_text_signal",
        benchmark="NPS e-com ngành thời trang median +30; <0 = toxic.",
        reference="Ghose & Ipeirotis (MIS Quarterly 2011); Reichheld (HBR 2003)."),
    "43_esg_risks": dict(
        title="ESG: Return Carbon Footprint + Deadstock Risk",
        act  ="ACT 11 — Text Mining & ESG (v3)",
        sources=["returns","shipments","inventory","products","order_items"],
        meaning="(a) CO₂ ước tính từ returns × 0.5 kg/return. "
                "(b) Deadstock = SKU age >180d & sell-through <30% — giá trị "
                "tồn kho có nguy cơ write-off.",
        pain="P17_esg_risk",
        benchmark="UK Environment Agency: 0.5 kg CO₂/return. "
                  "Deadstock >10% of inventory value = alarm.",
        reference="McKinsey State of Fashion 2024; BoF Sustainability Index."),

    # ---------------- ACT 12 — Decision Support (v4) ----------------
    "44_value_at_stake": dict(
        title="Value-at-Stake — Quantified Recommendations (M VND/year)",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["order_items","orders","customers","products",
                 "promotions","inventory","returns"],
        meaning="7 recommendations với annual uplift VND, investment, payback "
                "tính toán trực tiếp từ data (không phải opinion). "
                "Biến Prescriptive 7/10 → 10/10 theo McKinsey "
                "Value-at-Stake methodology.",
        pain="—",
        benchmark="Mỗi recommendation phải có $-impact, $-investment, "
                  "payback-months, risk rating.",
        reference="McKinsey Value-at-Stake; RICE framework (Intercom)."),
    "45_clv_churn": dict(
        title="Probabilistic CLV + Churn Score Distribution",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["customers","orders","order_items"],
        meaning="P(alive) per customer dùng exponential decay; expected 12m "
                "revenue = freq × 12 × P(alive) × AOV. Output: rank-list khách "
                "có giá trị tương lai cao nhất + phân phối churn.",
        pain="P5_customer_churn",
        benchmark="High-churn (score >0.7) target <30%; nếu cao hơn = retention "
                  "engine yếu.",
        reference="Fader & Hardie (2005) BG/NBD — simplified version."),
    "46_activation_funnel": dict(
        title="Activation Funnel — Signup → First Realized Order (3-way split)",
        act  ="ACT 6 — Industry-Grade Retail Metrics",
        sources=["customers","orders"],
        meaning="3-way split per consistency audit Finding 3: realized active "
                "(≥1 non-cancelled) vs cancelled-only (≥1 order, all cancelled — "
                "previously invisible) vs silent strict (0 orders). Phân tách theo "
                "acquisition_channel cho thấy channel-level activation rate là "
                "leading indicator của CAC efficiency, mạnh hơn raw signup count.",
        pain="P5_customer_churn",
        benchmark="B2C e-com activation within 30 days: 60-80% healthy "
                  "(Mixpanel Product Benchmarks 2023); <40% indicates acquisition-"
                  "quality issue.",
        reference="Andrew Chen activation funnel framework + cohort-style breakdown."),

    # ---------- ACT 12 — Diagnostic plots (v6 strategist follow-up) ----------
    "47_signup_gap_distribution": dict(
        title="Signup Gap Distribution — diagnostic for `signup_after_first_order`",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["customers","orders"],
        meaning="Histogram of first_order_date − signup_date in days. Localizes "
                "root cause of 89.34% violation: short median gap → guest-checkout "
                "retroactive registration; long → account migration; very long → "
                "simulator artifact.",
        pain="—",
        benchmark="Median gap of violators tells the story.",
        reference="Diagnostic — localizes data integrity finding."),
    "48_cancelled_only_profile": dict(
        title="Cancelled-only Segment Profile — fraud / COD-trap diagnostic",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["customers","orders","geography"],
        meaning="4-panel composite comparing 2,123 cancelled-only customers vs "
                "all customers across payment_method, acquisition_channel, region, "
                "and order_status mix. Red bars > blue = over-representation = "
                "fraud cluster or COD-trap signal.",
        pain="P16_wardrobing_fraud",
        benchmark="If COD share of segment >> overall (14.94%) → COD-suspect cluster.",
        reference="Profile-comparison diagnostic — Appriss Retail 2023 fraud playbook."),
    "49_activation_clv_overlay": dict(
        title="Activation Rate × Median CLV by Channel — CAC efficiency map",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["customers","orders","payments"],
        meaning="Bar = 30-day activation rate; overlay line = median CLV of "
                "realized active customers per channel. Reveals CAC efficiency: "
                "high-volume low-conversion channels burn cash even when CPL is "
                "low, while moderate-volume high-CLV channels can be best-ROI.",
        pain="P7_channel_attribution",
        benchmark="Healthy channel: ≥70% activation AND median CLV ≥ overall median.",
        reference="David Skok SaaS CAC efficiency framework adapted to retail."),
    "50_phantom_catalog_composition": dict(
        title="Phantom Catalog Composition — 814 unsold vs 1,598 sold SKUs",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["products","order_items"],
        meaning="3 panels: SKU split donut · phantom % per category · list-margin "
                "histogram (sold vs phantom). Phantom SKUs are systematically "
                "higher-margin than sold ones — explaining 10.8pp selection bias "
                "in chart 41 medians (catalog 30.6% → sold-only 19.8%).",
        pain="P11_unit_economics",
        benchmark="Healthy phantom rate <5% of catalog; >20% = catalog padding.",
        reference="Inditex assortment rationalization framework."),
    "51_scenario_projection": dict(
        title="Scenario Projection — Goods-only CM trajectory under operational levers",
        act  ="ACT 12 — Decision Support (v4)",
        sources=["order_items","products","orders","promotions"],
        meaning="4 scenarios A–D applied to baseline +2.26% goods-only CM: "
                "(A) activate 50% phantom catalog, (B) halve COD cancel rate, "
                "(C) kill bottom-quintile promos, (D) all combined. Each scenario "
                "shows ΔNR, new CM, Δpp, and the explicit assumption used.",
        pain="—",
        benchmark="Combined Scenario D should move CM toward >5% (industry minimum).",
        reference="McKinsey Value-at-Stake; what-if sensitivity standard practice."),
}


# =============================================================================
# 5. HELPERS
# =============================================================================
def export_fig(fig, name: str, width: int = PNG_WIDTH):
    h = fig.layout.height or 500
    path = FIG_DIR / f"{name}.png"
    fig.write_image(path, width=width, height=h, scale=PNG_SCALE)
    print(f"   ✓ {path.name}")


def gini_coefficient(values: np.ndarray) -> float:
    """Gini of a non-negative distribution. 0 = equal, 1 = total inequality."""
    v = np.asarray(values, dtype=float)
    v = np.sort(v[v >= 0])
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def kaplan_meier(durations: np.ndarray, events: np.ndarray):
    """Naive KM estimator. durations=time to event/censor, events=1 if event."""
    df = pd.DataFrame({"t": durations, "e": events}).sort_values("t")
    s, out = 1.0, []
    for t, g in df.groupby("t"):
        at_risk = (df["t"] >= t).sum()
        d = int(g["e"].sum())
        if at_risk > 0:
            s = s * (1 - d / at_risk)
        out.append((t, s))
    return pd.DataFrame(out, columns=["t", "survival"])


# =============================================================================
# 6. DATA LOADING
# =============================================================================
def load_raw():
    files = {
        "products":"products.csv", "customers":"customers.csv",
        "promotions":"promotions.csv", "geography":"geography.csv",
        "orders":"orders.csv", "order_items":"order_items.csv",
        "payments":"payments.csv", "shipments":"shipments.csv",
        "returns":"returns.csv", "reviews":"reviews.csv",
        "sales":"sales.csv", "inventory":"inventory.csv",
        "web_traffic":"web_traffic.csv",
    }
    date_cols = {
        "customers":["signup_date"], "promotions":["start_date","end_date"],
        "orders":["order_date"], "shipments":["ship_date","delivery_date"],
        "returns":["return_date"], "reviews":["review_date"],
        "sales":["Date"], "inventory":["snapshot_date"], "web_traffic":["date"],
    }
    data = {}
    for key, fname in files.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=date_cols.get(key, []))
        data[key] = df
        print(f"  loaded {fname:<22} shape={df.shape}")
    return data


# =============================================================================
# 7. ANALYTICAL BASE TABLES
# =============================================================================
def build_tx_master(d):
    """Per [unit_price_interpretation]=A: revenue formula = qty × unit_price.
       Per [A3_realized_order_def]: filter status ∈ {delivered, returned}.
       Per [A8_time_periods]: tag period column for COVID-aware splits.
       Per [oi_composite_pk]: 16 duplicates in (order_id, product_id) — dedup before join."""

    # Per [A14_discount_math]: only 36% promo lines reconcile —
    # use products.price as authoritative list price (column already merged below).
    # Per [A15_stacked_promo_frequency]: promo_id_2 = 0.03% lines, ignore.

    # Per [oi_composite_pk]: aggregate 16 duplicate lines so 1 row per (order, product)
    oi = (d["order_items"]
            .groupby(["order_id","product_id"], as_index=False)
            .agg(quantity=("quantity","sum"),
                 unit_price=("unit_price","mean"),
                 discount_amount=("discount_amount","sum"),
                 promo_id=("promo_id","first"),
                 promo_id_2=("promo_id_2","first")))

    tx = oi.merge(d["orders"], on="order_id", how="left")
    tx = tx.merge(d["products"], on="product_id", how="left", suffixes=("","_prod"))
    tx = tx.merge(d["customers"][["customer_id","gender","age_group",
                                  "acquisition_channel","signup_date"]],
                  on="customer_id", how="left")
    tx = tx.merge(d["geography"], on="zip", how="left", suffixes=("","_geo"))

    # Money — Per [A5_revenue_levels] taxonomy
    # Net = qty × unit_price (after promo); Gross = qty × products.price (list)
    tx["net_revenue"]   = tx["quantity"] * tx["unit_price"]                    # AFTER promo
    tx["gross_revenue"] = tx["quantity"] * tx["price"]                         # LIST price
    tx["cogs_total"]    = tx["quantity"] * tx["cogs"]
    tx["gross_profit"]  = tx["net_revenue"] - tx["cogs_total"]
    tx["margin_pct"]    = np.where(tx["net_revenue"] > 0,
                                   tx["gross_profit"]/tx["net_revenue"], np.nan)

    tx["year"]   = tx["order_date"].dt.year
    tx["month"]  = tx["order_date"].dt.month
    tx["ym"]     = tx["order_date"].dt.to_period("M").dt.to_timestamp()
    tx["dow"]    = tx["order_date"].dt.day_name()

    # Per [A8_time_periods] — COVID-aware splits
    def _period(d):
        if d.year < 2020:                  return "pre_covid"
        if d.year < 2022:                  return "covid"
        return "post_covid"
    tx["covid_period"] = tx["order_date"].apply(_period)

    tx["is_promo"]      = tx["promo_id"].notna().astype(int)
    tx["discount_pct"]  = (tx["price"] - tx["unit_price"]) / tx["price"].clip(lower=1)
    tx["is_full_price"] = (tx["discount_pct"] < 0.001).astype(int)
    tx["is_cancelled"]  = (tx["order_status"] == "cancelled").astype(int)

    ret = d["returns"].groupby(["order_id","product_id"], as_index=False).agg(
        return_qty=("return_quantity","sum"),
        refund_amt=("refund_amount","sum"))
    tx = tx.merge(ret, on=["order_id","product_id"], how="left")
    tx["return_qty"]  = tx["return_qty"].fillna(0)
    tx["refund_amt"]  = tx["refund_amt"].fillna(0)
    tx["is_returned"] = (tx["return_qty"] > 0).astype(int)

    # Per [A3_realized_order_def] + [order_status_in_oi]:
    # cancelled orders DO have order_items rows (59,462 in this dataset).
    # We KEEP all rows in tx but downstream charts must filter is_cancelled=0
    # for revenue computation. The flag let us also analyze cancellation flow.
    return tx


def build_customer_master(d, tx):
    """Per [A4_active_customer_def]: ≥1 delivered order = active.
       Per [silent_cohort]: 31,684 customers (26%) signed up but never bought.
       This ABT covers ACTIVE only; build_silent_cohort() returns the rest."""

    # Use only realized orders for RFM (no cancelled, since they aren't real revenue)
    tx_real = tx[tx["is_cancelled"] == 0]
    snap = tx_real["order_date"].max() + pd.Timedelta(days=1)

    agg = tx_real.groupby("customer_id").agg(
        first_order=("order_date","min"), last_order=("order_date","max"),
        orders=("order_id","nunique"), units=("quantity","sum"),
        revenue=("net_revenue","sum"), profit=("gross_profit","sum"),
        categories=("category", lambda s: s.nunique()),
        return_rate=("is_returned","mean"),
        full_price_share=("is_full_price","mean"),
    ).reset_index()
    agg["recency_days"] = (snap - agg["last_order"]).dt.days
    agg["tenure_days"]  = (agg["last_order"] - agg["first_order"]).dt.days
    agg["aov"]          = agg["revenue"] / agg["orders"].clip(lower=1)

    for col, label, ascending in [("recency_days","R",False),
                                   ("orders","F",True),
                                   ("revenue","M",True)]:
        agg[label] = pd.qcut(agg[col].rank(method="first"), 5,
                             labels=[1,2,3,4,5]).astype(int)
        if not ascending:
            agg[label] = 6 - agg[label]

    def segment(r):
        R, F, M = r["R"], r["F"], r["M"]
        if R >= 4 and F >= 4 and M >= 4: return "Champion"
        if R >= 3 and F >= 3:            return "Loyal"
        if R >= 4 and F <= 2:            return "Potential"
        if R <= 2 and F >= 3:            return "At-Risk"
        if R <= 2 and F <= 2 and M >= 3: return "Hibernating"
        return "Lost"
    agg["rfm_segment"] = agg.apply(segment, axis=1)

    agg = agg.merge(d["customers"][["customer_id","gender","age_group",
                                     "acquisition_channel","signup_date","zip"]],
                    on="customer_id", how="left")
    # Per [signup_after_first_order]: 80,623 customers (89%) have signup_date AFTER
    # their first order — signup_date is NOT reliable for cohort/tenure analysis.
    # Use FIRST-ORDER month as cohort anchor instead.
    agg["activation_cohort"] = agg["first_order"].dt.to_period("M").dt.to_timestamp()
    agg["signup_cohort"]     = agg["signup_date"].dt.to_period("M").dt.to_timestamp()  # kept for reference only
    agg["signup_after_first_order"] = (agg["signup_date"] > agg["first_order"]).astype(int)
    # days_to_first_ord is unreliable when signup > first_order; report as informational
    agg["days_to_first_ord"] = (agg["first_order"] - agg["signup_date"]).dt.days
    return agg


def build_silent_cohort(d, tx):
    """Per [silent_cohort]: customers who signed up but never placed any order.
       Returns a DataFrame mirroring customer_abt's metadata columns + a flag."""
    active_ids = set(tx["customer_id"].unique())
    silent = d["customers"][~d["customers"]["customer_id"].isin(active_ids)].copy()
    silent["signup_cohort"] = silent["signup_date"].dt.to_period("M").dt.to_timestamp()
    silent["rfm_segment"]   = "Silent (never ordered)"
    silent["orders"]        = 0
    silent["revenue"]       = 0.0
    return silent


def build_time_master(d, tx):
    """Per [A3_realized_order_def]: only realized orders count for revenue.
       Per [A8_time_periods]: tag covid_period column.
       Per [A13_sales_csv_role]: keeps sales.csv as ground truth (joined below)."""

    tx_real = tx[tx["is_cancelled"] == 0]    # filter cancelled per data contract
    daily = tx_real.groupby("order_date").agg(
        revenue=("net_revenue","sum"), cogs=("cogs_total","sum"),
        orders=("order_id","nunique"), units=("quantity","sum"),
        promo_orders=("is_promo","sum")).reset_index().rename(
        columns={"order_date":"date"})
    tm = daily.merge(d["web_traffic"].groupby("date", as_index=False).agg(
        sessions=("sessions","sum"), visitors=("unique_visitors","sum"),
        page_views=("page_views","sum"), bounce=("bounce_rate","mean")),
                     on="date", how="left")
    tm["gross_profit"] = tm["revenue"] - tm["cogs"]
    tm["aov"]          = tm["revenue"] / tm["orders"].clip(lower=1)
    tm["conv_rate"]    = tm["orders"]  / tm["sessions"].clip(lower=1)
    tm["dow"]   = tm["date"].dt.day_name()
    tm["month"] = tm["date"].dt.month
    tm["year"]  = tm["date"].dt.year
    tm["ym"]    = tm["date"].dt.to_period("M").dt.to_timestamp()

    # Per [A8_time_periods] — COVID-aware split
    def _period(d):
        if d.year < 2020:                  return "pre_covid"
        if d.year < 2022:                  return "covid"
        return "post_covid"
    tm["covid_period"] = tm["date"].apply(_period)
    return tm


# =============================================================================
# 8. HEADLINE METRICS
# =============================================================================
def compute_metrics(tx, cust, tm, d):
    m = {}

    # Per [A3_realized_order_def] + [order_status_in_oi]:
    # Headline KPIs use REALIZED revenue only (filter cancelled).
    tx_real = tx[tx["is_cancelled"] == 0]

    m["period_start"]   = tx["order_date"].min()
    m["period_end"]     = tx["order_date"].max()

    # Per [silent_cohort] + consistency audit — 3-way customer split:
    #   realized_active  = ≥1 NON-cancelled order (per [A4_active_customer_def])
    #   cancelled_only   = ≥1 order, ALL cancelled (invisible in v3-v5; flagged in v6 audit)
    #   silent_strict    = 0 orders ever (true silent per assumption description)
    #   sum = total signups
    n_total              = d["customers"].shape[0]
    n_any_order          = d["orders"]["customer_id"].nunique()
    n_realized_active    = cust.shape[0]              # active per [A4]
    m["n_customers_total"]   = n_total
    m["n_customers"]         = n_realized_active       # primary "active" metric
    m["n_realized_active"]   = n_realized_active
    m["n_any_order"]         = n_any_order
    m["n_cancelled_only"]    = n_any_order - n_realized_active
    m["n_silent_strict"]     = n_total - n_any_order
    m["n_silent"]            = m["n_silent_strict"]    # alias matching assumption description
    m["activation_rate"]         = n_realized_active / n_total   # realized = primary
    m["activation_rate_strict"]  = n_any_order / n_total          # ≥1 any order
    m["activation_rate_realized"]= n_realized_active / n_total

    m["n_orders_all"]   = tx["order_id"].nunique()                  # incl. cancelled
    m["n_orders"]       = tx_real["order_id"].nunique()              # realized only
    m["n_cancelled"]    = (tx[tx["is_cancelled"]==1]["order_id"].nunique())
    m["cancel_rate"]    = m["n_cancelled"] / max(m["n_orders_all"], 1)
    m["n_products"]     = d["products"].shape[0]
    m["n_promos"]       = d["promotions"].shape[0]

    m["gross_revenue"]  = tx_real["gross_revenue"].sum()       # qty × list price (Per [A5])
    m["net_revenue"]    = tx_real["net_revenue"].sum()         # qty × unit_price (Per [A5])
    m["total_discount"] = (m["gross_revenue"] - m["net_revenue"])
    m["total_cogs"]     = tx_real["cogs_total"].sum()
    m["gross_profit"]   = tx_real["gross_profit"].sum()
    m["gross_margin"]   = m["gross_profit"] / m["net_revenue"]
    m["aov"]            = m["net_revenue"] / m["n_orders"]
    m["return_rate"]    = tx_real["is_returned"].mean()
    m["total_refund"]   = tx_real["refund_amt"].sum()
    m["repeat_rate"]    = (cust["orders"] > 1).mean()
    m["avg_orders_per_cust"] = cust["orders"].mean()
    m["avg_revenue_per_cust"]= cust["revenue"].mean()
    m["median_recency"] = cust["recency_days"].median()
    m["segment_counts"] = cust["rfm_segment"].value_counts().to_dict()
    m["promo_share"]    = tx_real["is_promo"].mean()
    m["full_price_share"]    = tx_real["is_full_price"].mean()
    m["full_price_rev_share"]= (tx_real.loc[tx_real.is_full_price==1,"net_revenue"].sum()
                                 / m["net_revenue"])

    # Concentration metrics — use realized only
    sku_rev = tx_real.groupby("product_id")["net_revenue"].sum()
    cust_rev= cust["revenue"]
    m["sku_gini"]   = gini_coefficient(sku_rev.values)
    m["cust_gini"]  = gini_coefficient(cust_rev.values)
    cum = sku_rev.sort_values(ascending=False).cumsum() / sku_rev.sum()
    m["sku_share_80"] = float((cum >= 0.80).idxmax() and
                              (cum.values >= 0.80).argmax() + 1) / len(sku_rev)
    cum_c = cust_rev.sort_values(ascending=False).cumsum() / cust_rev.sum()
    m["top10pct_cust_rev_share"] = float(
        cum_c.iloc[: max(1, int(len(cust_rev)*0.10))].iloc[-1])

    # Benchmark gap
    m["vs_return_bm"]   = m["return_rate"] - BENCHMARKS["return_rate_online_fashion"]
    m["vs_repeat_bm"]   = m["repeat_rate"] - BENCHMARKS["repeat_rate_12m"]
    m["vs_margin_bm"]   = m["gross_margin"] - BENCHMARKS["gross_margin_fashion"]

    # ---- v3: Contribution margin (strategist Gap #1) ----
    # Per [shipping_fee_semantics]: dataset shipping_fee = customer-side fee,
    # NOT carrier cost. CM here is "as-data" — must be reported alongside
    # sensitivity scenarios.
    vc = _estimate_variable_costs(tx_real, d)
    # Per [A9_shipping_treatment]: EXCLUDE shipping from CM entirely.
    # Carrier cost not in dataset; customer-paid fee is pass-through. Goods-only CM.
    m["shipping_cost"]   = vc["shipping"]            # kept as informational only
    m["payment_fee"]     = vc["payment"]
    m["packaging_cost"]  = vc["packaging"]
    m["return_proc_cost"]= vc["return_proc"]
    non_ship = vc["payment"] + vc["packaging"] + vc["return_proc"] + vc["refunds"]
    m["contribution_margin"] = m["gross_profit"] - non_ship    # goods-only per A9
    m["contribution_margin_pct"] = m["contribution_margin"] / m["net_revenue"]

    # Shipping fee data quality flag (informational; not in CM)
    sf_series = d["shipments"]["shipping_fee"].fillna(0)
    m["shipping_fee_zero_pct"] = float((sf_series == 0).mean())
    m["shipping_fee_pct_revenue"] = vc["shipping"] / m["net_revenue"]

    # ---- v3: COD metrics ----
    orders = d["orders"]
    m["cod_share"]          = (orders["payment_method"] == "cod").mean()
    m["cancel_rate_overall"]= (orders["order_status"] == "cancelled").mean()
    cod_orders = orders[orders["payment_method"] == "cod"]
    m["cancel_rate_cod"] = ((cod_orders["order_status"] == "cancelled").mean()
                            if len(cod_orders) > 0 else 0)

    # ---- v3: Price-COGS paradox ----
    prods = d["products"]
    m["sku_negative_margin_pct"] = ((prods["cogs"] >= prods["price"]).mean()
                                     if "price" in prods.columns else 0)
    m["median_list_margin_pct"]  = ((1 - prods["cogs"]/prods["price"]).median()
                                     if "price" in prods.columns else np.nan)

    # ---- v6: Signup integrity stats (Finding 4) ----
    first_ord = d["orders"].groupby("customer_id")["order_date"].min()
    cu_join = d["customers"].set_index("customer_id").join(
        first_ord.rename("first_order"), how="inner")
    m["n_signup_after_first_order"] = int((cu_join["signup_date"] > cu_join["first_order"]).sum())
    m["signup_after_first_order_pct"] = float(
        (cu_join["signup_date"] > cu_join["first_order"]).mean())

    # ---- v4: Data Quality summary stats (used in MD report) ----
    dq = compute_dq_stats(d)
    m["dq_top_nulls"] = (dq["nulls_df"][dq["nulls_df"]["null_pct"] > 0]
                              .nlargest(8, "null_pct")
                              .to_dict(orient="records"))
    m["dq_integrity"] = dq["integrity"].to_dict(orient="records")
    m["dq_shipping"]  = dq["shipping"]
    m["dq_date_cov"]  = dq["date_cov"].to_dict(orient="records")
    return m


# =============================================================================
# 9. CHART 00 — DATA QUALITY AUDIT
# =============================================================================
def compute_dq_stats(d):
    """Return DQ stats dict (used by chart_00 + MD report)."""
    rows = []
    for tbl, df in d.items():
        for col in df.columns:
            rows.append(dict(table=tbl, col=col,
                             null_pct=df[col].isna().mean()*100,
                             rows=len(df)))
    nd = pd.DataFrame(rows)

    # Date coverage
    date_cov = []
    for tbl, dcol in [("orders","order_date"),("shipments","ship_date"),
                      ("returns","return_date"),("reviews","review_date"),
                      ("sales","Date"),("inventory","snapshot_date"),
                      ("web_traffic","date"),("customers","signup_date"),
                      ("promotions","start_date")]:
        df = d.get(tbl)
        if df is not None and dcol in df.columns and df[dcol].notna().any():
            date_cov.append(dict(table=tbl,
                                 start=df[dcol].min(),
                                 end=df[dcol].max()))
    dc = pd.DataFrame(date_cov)

    # Referential integrity — count orphan rows per FK
    integrity = []

    def orphan(child_df, child_col, parent_df, parent_col):
        return int((~child_df[child_col].isin(parent_df[parent_col])).sum())

    integrity.append(dict(check="order_items → orders.order_id",
                          orphan_count=orphan(d["order_items"],"order_id",
                                              d["orders"],"order_id")))
    integrity.append(dict(check="order_items → products.product_id",
                          orphan_count=orphan(d["order_items"],"product_id",
                                              d["products"],"product_id")))
    integrity.append(dict(check="orders → customers.customer_id",
                          orphan_count=orphan(d["orders"],"customer_id",
                                              d["customers"],"customer_id")))
    integrity.append(dict(check="returns → orders.order_id",
                          orphan_count=orphan(d["returns"],"order_id",
                                              d["orders"],"order_id")))
    integrity.append(dict(check="reviews → orders.order_id",
                          orphan_count=orphan(d["reviews"],"order_id",
                                              d["orders"],"order_id")))
    integrity.append(dict(check="customers.zip → geography.zip",
                          orphan_count=orphan(d["customers"],"zip",
                                              d["geography"],"zip")))
    integ_df = pd.DataFrame(integrity)

    # CRITICAL: shipping_fee diagnostic — feeds the unit-economics paradox
    sf = d["shipments"]["shipping_fee"].fillna(0)
    sf_stats = dict(
        zero_pct  = float((sf == 0).mean() * 100),
        null_pct  = float(d["shipments"]["shipping_fee"].isna().mean() * 100),
        mean_fee  = float(sf.mean()),
        median_fee= float(sf.median()),
        total_fee = float(sf.sum()),
        n_shipments = int(len(sf)),
    )

    return dict(nulls_df=nd, date_cov=dc, integrity=integ_df, shipping=sf_stats)


def chart_00_data_quality(d):
    """Substantive DQ: nulls, date coverage, ref-integrity, shipping diagnostic."""
    s = compute_dq_stats(d)
    nd, dc, integ_df, sf = s["nulls_df"], s["date_cov"], s["integrity"], s["shipping"]

    fig = make_subplots(rows=2, cols=2,
                        column_widths=[0.55, 0.45],
                        row_heights=[0.5, 0.5],
                        subplot_titles=[
                            "Null % by column — top-15 with non-zero null",
                            "Date coverage by table",
                            "Referential integrity (orphan FK counts)",
                            f"Shipping-fee distribution ⚠  "
                            f"{sf['zero_pct']:.1f}% are 0 VND, "
                            f"mean = {sf['mean_fee']:,.0f} VND"])

    # Panel 1: top-15 null %
    sub = nd[nd.null_pct > 0].nlargest(15, "null_pct")
    if len(sub) == 0:
        sub = pd.DataFrame([dict(table="—", col="(no nulls detected)",
                                 null_pct=0, rows=0)])
    fig.add_trace(go.Bar(x=sub["null_pct"],
                         y=sub["table"] + " · " + sub["col"],
                         orientation="h",
                         marker=dict(color=sub["null_pct"], colorscale="Reds"),
                         text=sub["null_pct"].round(1).astype(str) + "%",
                         textposition="outside",
                         showlegend=False),
                  row=1, col=1)

    # Panel 2: date coverage
    for _, r in dc.iterrows():
        fig.add_trace(go.Scatter(x=[r["start"], r["end"]],
                                 y=[r["table"]]*2,
                                 mode="lines+markers",
                                 line=dict(color=PALETTE["primary"], width=6),
                                 showlegend=False), row=1, col=2)

    # Panel 3: referential integrity
    colors = [PALETTE["primary"] if c == 0 else PALETTE["danger"]
              for c in integ_df["orphan_count"]]
    fig.add_trace(go.Bar(x=integ_df["orphan_count"], y=integ_df["check"],
                         orientation="h", marker_color=colors,
                         text=integ_df["orphan_count"], textposition="outside",
                         showlegend=False),
                  row=2, col=1)

    # Panel 4: shipping_fee distribution — flag the unit-economics issue
    sf_series = d["shipments"]["shipping_fee"].fillna(0)
    fig.add_trace(go.Histogram(x=sf_series, nbinsx=50,
                               marker_color=PALETTE["accent"],
                               showlegend=False), row=2, col=2)
    fig.add_vline(x=sf['mean_fee'], line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"mean {sf['mean_fee']:,.0f}",
                  row=2, col=2)

    fig.update_xaxes(title_text="% null",          row=1, col=1)
    fig.update_xaxes(title_text="orphan rows",     row=2, col=1)
    fig.update_xaxes(title_text="shipping fee VND",row=2, col=2)
    fig.update_layout(title="<b>Data Quality Audit</b> — actual results",
                      height=820)
    export_fig(fig, "00_data_quality")
    return fig


# =============================================================================
# 10. ACT 1 — BUSINESS PULSE
# =============================================================================
def chart_01_kpi_scorecard(tx, cust, tm):
    revenue     = tx["net_revenue"].sum()
    orders      = tx["order_id"].nunique()
    aov         = revenue / orders
    gross_mgn   = tx["gross_profit"].sum() / revenue
    return_rate = tx["is_returned"].mean()
    repeat_rate = (cust["orders"] > 1).mean()

    fig = make_subplots(rows=1, cols=6,
                        specs=[[{"type":"indicator"}]*6],
                        subplot_titles=["Revenue","Orders","AOV",
                                        "Gross Margin","Return Rate","Repeat Rate"])
    metrics = [(revenue,"$,.0f"),(orders,",.0f"),(aov,"$,.0f"),
               (gross_mgn,".1%"),(return_rate,".1%"),(repeat_rate,".1%")]
    for i,(val,fmt) in enumerate(metrics, start=1):
        fig.add_trace(go.Indicator(mode="number", value=val,
                                   number={"valueformat":fmt,
                                           "font":{"size":34,
                                                   "color":PALETTE["primary"]}}),
                      row=1, col=i)
    fig.update_layout(title="<b>Executive KPI Scorecard</b> — Lifetime snapshot",
                      height=260, margin=dict(t=70,b=10))
    export_fig(fig, "01_kpi_scorecard")
    return fig


def chart_02_revenue_waterfall(tx):
    gross  = (tx["quantity"]*tx["unit_price"]).sum() + tx["discount_amount"].sum()
    disc   = -tx["discount_amount"].sum()
    net    =  tx["net_revenue"].sum()
    cogs   = -tx["cogs_total"].sum()
    refund = -tx["refund_amt"].sum()
    profit = net + cogs + refund
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","total","relative","relative","total"],
        x=["Gross Revenue","Promo Discounts","Net Revenue","COGS","Refunds","Gross Profit"],
        y=[gross, disc, net, cogs, refund, profit],
        connector={"line":{"color":PALETTE["muted"]}},
        decreasing={"marker":{"color":PALETTE["danger"]}},
        increasing={"marker":{"color":PALETTE["primary"]}},
        totals    ={"marker":{"color":PALETTE["accent"]}},
        text=[f"{v:,.0f}" for v in [gross,disc,net,cogs,refund,profit]],
        textposition="outside"))
    fig.update_layout(title="<b>Revenue Waterfall</b> — Từ top-line xuống gross profit",
                      yaxis_title="VND", height=520)
    export_fig(fig, "02_revenue_waterfall")
    return fig


def chart_03_revenue_timeline_stl(tm, promos):
    s = tm.set_index("date")["revenue"].asfreq("D").interpolate()
    stl = STL(s, period=365, robust=True).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed","Trend","Seasonal","Residual"],
                        vertical_spacing=0.04)
    for i, series in enumerate([stl.observed, stl.trend, stl.seasonal, stl.resid], start=1):
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                 line=dict(width=1.2, color=PALETTE["primary"])),
                      row=i, col=1)
    top = (promos.assign(span=(promos["end_date"]-promos["start_date"]).dt.days)
                  .sort_values("span", ascending=False).head(5))
    for _, p in top.iterrows():
        fig.add_vrect(x0=p["start_date"], x1=p["end_date"],
                      fillcolor=PALETTE["accent"], opacity=0.12, line_width=0,
                      row=1, col=1,
                      annotation_text=p["promo_name"], annotation_position="top left")
    fig.update_layout(height=780, showlegend=False,
                      title="<b>Revenue Timeline</b> — STL decomposition + promo overlays")
    export_fig(fig, "03_revenue_stl")
    return fig


def chart_04_yoy_heatmap(tm):
    g = (tm.groupby([tm["year"], tm["month"]])["revenue"].sum()
           .unstack().sort_index())
    yoy = g.pct_change() * 100
    fig = go.Figure(go.Heatmap(z=yoy.values,
                               x=[f"M{m:02d}" for m in yoy.columns], y=yoy.index,
                               colorscale="RdYlGn", zmid=0,
                               text=np.round(yoy.values,1),
                               texttemplate="%{text}%",
                               colorbar=dict(title="YoY %")))
    fig.update_layout(title="<b>Year × Month YoY Growth</b>",
                      xaxis_title="Month", yaxis_title="Year", height=500)
    export_fig(fig, "04_yoy_heatmap")
    return fig


def chart_05_dow_month_heatmap(tm):
    dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    piv = (tm.groupby(["dow","month"])["revenue"].mean()
             .unstack().reindex(dow))
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale="Greens"))
    fig.update_layout(title="<b>Day-of-Week × Month</b> — avg daily revenue",
                      xaxis_title="Month", yaxis_title="", height=420)
    export_fig(fig, "05_dow_month")
    return fig


# =============================================================================
# 11. ACT 2 — CUSTOMERS
# =============================================================================
def chart_06_rfm_scatter(cust):
    fig = px.scatter(cust, x="recency_days", y="orders", size="revenue",
                     color="rfm_segment",
                     color_discrete_map=PALETTE["seg"],
                     hover_data=["customer_id","aov","tenure_days"],
                     log_y=True, size_max=40,
                     title="<b>RFM Segmentation</b> — 6-segment customer map")
    fig.update_layout(xaxis_title="Recency (days since last order)",
                      yaxis_title="Frequency (orders, log)", height=600)
    export_fig(fig, "06_rfm_scatter")
    return fig


def chart_07_cohort_heatmap(cust, tx):
    """Per [signup_after_first_order]: signup_date unreliable for 89% of customers.
       Use FIRST-ORDER month (`activation_cohort`) as the cohort anchor instead."""
    base = tx[tx["is_cancelled"]==0][["customer_id","order_date"]].merge(
        cust[["customer_id","activation_cohort"]], on="customer_id")
    base["months_since"] = ((base["order_date"].dt.year - base["activation_cohort"].dt.year)*12
                            + base["order_date"].dt.month - base["activation_cohort"].dt.month)
    cohort_sizes = cust.groupby("activation_cohort")["customer_id"].nunique()
    active = (base.groupby(["activation_cohort","months_since"])["customer_id"]
                  .nunique().unstack(fill_value=0))
    retention = active.div(cohort_sizes, axis=0) * 100
    retention = retention.iloc[:, :24]
    fig = go.Figure(go.Heatmap(z=retention.values, x=retention.columns,
                               y=retention.index.strftime("%Y-%m"),
                               colorscale="Greens", zmin=0,
                               zmax=retention.values.max(),
                               colorbar=dict(title="% active")))
    fig.update_layout(
        title="<b>Cohort Retention Heatmap</b> — anchor = first-order month "
              "(signup_date unreliable per data audit)",
        xaxis_title="Months since FIRST ORDER",
        yaxis_title="Activation cohort", height=800)
    export_fig(fig, "07_cohort_retention")
    return fig


def chart_08_clv_by_channel(cust):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=["CLV distribution","Median CLV rank"])
    for ch, sub in cust.groupby("acquisition_channel"):
        fig.add_trace(go.Box(y=sub["revenue"], name=str(ch), boxmean=True,
                             marker_color=PALETTE["primary"]), row=1, col=1)
    rank = (cust.groupby("acquisition_channel")["revenue"].median()
                 .sort_values(ascending=True))
    fig.add_trace(go.Bar(x=rank.values, y=rank.index, orientation="h",
                         marker_color=PALETTE["accent"]), row=1, col=2)
    fig.update_layout(title="<b>CLV by Acquisition Channel</b>",
                      showlegend=False, height=540)
    fig.update_yaxes(type="log", row=1, col=1)
    export_fig(fig, "08_clv_channel")
    return fig


def chart_09_demographic_sunburst(tx):
    sb = (tx.dropna(subset=["gender","age_group"])
            .groupby(["gender","age_group","category","segment"])["net_revenue"]
            .sum().reset_index())
    fig = px.sunburst(sb, path=["gender","age_group","category","segment"],
                      values="net_revenue",
                      color="net_revenue", color_continuous_scale="Greens",
                      title="<b>Demographic × Product Revenue Sunburst</b>")
    fig.update_layout(height=700)
    export_fig(fig, "09_demographic_sunburst")
    return fig


def chart_10_geo_map(tx):
    geo_rev = (tx.groupby(["region","city"])
                 .agg(revenue=("net_revenue","sum"),
                      orders=("order_id","nunique"),
                      customers=("customer_id","nunique")).reset_index())
    geo_rev["rev_per_customer"] = geo_rev["revenue"] / geo_rev["customers"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"type":"domain"}, {"type":"xy"}]],
                        subplot_titles=["Revenue by region (treemap)",
                                        "Top-15 cities — Rev / Customer"])
    tm_ = px.treemap(geo_rev, path=["region","city"], values="revenue",
                     color="rev_per_customer", color_continuous_scale="Greens")
    for tr in tm_.data: fig.add_trace(tr, row=1, col=1)
    top15 = geo_rev.nlargest(15, "rev_per_customer")
    fig.add_trace(go.Bar(x=top15["rev_per_customer"], y=top15["city"],
                         orientation="h", marker_color=PALETTE["primary"]),
                  row=1, col=2)
    fig.update_layout(title="<b>Geographic Revenue Distribution</b>",
                      height=600, showlegend=False)
    export_fig(fig, "10_geo_map")
    return fig


# =============================================================================
# 12. ACT 3 — REVENUE DRIVERS
# =============================================================================
def chart_11_bcg_matrix(tx):
    cy = tx["year"].max(); py = cy - 1
    cur = tx[tx["year"]==cy].groupby("category")["net_revenue"].sum()
    prv = tx[tx["year"]==py].groupby("category")["net_revenue"].sum()
    growth = ((cur - prv) / prv * 100).rename("yoy_growth")
    share  = (cur / cur.sum() * 100).rename("share")
    profit = tx[tx["year"]==cy].groupby("category")["gross_profit"].sum()
    bcg = pd.concat([growth, share, profit], axis=1).reset_index().dropna()
    fig = px.scatter(bcg, x="share", y="yoy_growth", size="gross_profit",
                     text="category", size_max=80,
                     color="yoy_growth", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0)
    fig.add_hline(y=bcg["yoy_growth"].median(), line_dash="dash",
                  line_color=PALETTE["muted"])
    fig.add_vline(x=bcg["share"].median(), line_dash="dash",
                  line_color=PALETTE["muted"])
    fig.update_layout(title="<b>BCG Growth-Share Matrix</b> — Category portfolio",
                      xaxis_title="Market share (% of revenue)",
                      yaxis_title="YoY growth (%)", height=640)
    fig.update_traces(textposition="top center")
    export_fig(fig, "11_bcg_matrix")
    return fig


def chart_12_price_elasticity(tx):
    t = tx.copy()
    t["disc_bin"] = pd.cut(t["discount_pct"],
                           bins=[-0.01,0,0.1,0.2,0.3,0.5,1.01],
                           labels=["0%","1-10%","10-20%","20-30%","30-50%",">50%"])
    el = (t.groupby(["segment","disc_bin"], observed=True)
            .agg(units=("quantity","mean"),
                 orders=("order_id","nunique")).reset_index())
    fig = px.line(el, x="disc_bin", y="units", color="segment", markers=True,
                  title="<b>Price Elasticity</b> — Avg units per line by discount depth")
    fig.update_layout(height=520,
                      xaxis_title="Discount depth",
                      yaxis_title="Avg units / order-line")
    export_fig(fig, "12_price_elasticity")
    return fig


def chart_13_promo_roi_bubble(tx, promos):
    p = (tx[tx["promo_id"].notna()]
         .groupby("promo_id").agg(
             discount=("discount_amount","sum"),
             revenue=("net_revenue","sum"),
             orders=("order_id","nunique"),
             units=("quantity","sum")).reset_index())
    p = p.merge(promos[["promo_id","promo_name","promo_type","applicable_category"]],
                on="promo_id", how="left")
    p["roi"] = p["revenue"] / p["discount"].clip(lower=1)
    fig = px.scatter(p, x="discount", y="revenue", size="orders",
                     color="roi", color_continuous_scale="RdYlGn",
                     hover_data=["promo_name","applicable_category","roi"],
                     log_x=True, log_y=True, size_max=55,
                     title="<b>Promotion ROI Landscape</b>")
    fig.update_layout(height=600, xaxis_title="Discount given (log)",
                      yaxis_title="Revenue generated (log)")
    export_fig(fig, "13_promo_roi")
    return fig


def chart_14_size_color_returns(tx):
    piv = (tx.groupby(["category","size"])["is_returned"].mean().unstack() * 100)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale="Reds",
                               text=np.round(piv.values,1),
                               texttemplate="%{text}%"))
    # Benchmark annotation
    fig.add_annotation(x=0, y=-1, showarrow=False,
                       text=f"Industry avg return rate: "
                            f"{BENCHMARKS['return_rate_online_fashion']:.0%}",
                       font=dict(color=PALETTE["muted"]))
    fig.update_layout(title="<b>Return-rate heatmap</b> — Category × Size",
                      height=480)
    export_fig(fig, "14_return_heatmap")
    return fig


def chart_15_copurchase_network(tx, top_n=30):
    baskets = tx.groupby("order_id")["category"].apply(set)
    pairs = Counter()
    for items in baskets:
        for a, b in combinations(sorted(items), 2):
            pairs[(a, b)] += 1
    top = pd.DataFrame(pairs.most_common(top_n), columns=["pair","count"])
    top[["cat_a","cat_b"]] = pd.DataFrame(top["pair"].tolist(), index=top.index)
    import networkx as nx
    G = nx.Graph()
    for _, r in top.iterrows():
        G.add_edge(r["cat_a"], r["cat_b"], weight=r["count"])
    pos = nx.spring_layout(G, seed=RANDOM_SEED)
    edge_x, edge_y = [], []
    for (a, b, _) in G.edges(data=True):
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color=PALETTE["muted"], width=1),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             marker=dict(size=24, color=PALETTE["primary"]),
                             text=list(G.nodes()), textposition="top center"))
    fig.update_layout(title="<b>Co-purchase Network</b> — Category affinity",
                      showlegend=False, height=600,
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False))
    export_fig(fig, "15_copurchase_network")
    return fig


# =============================================================================
# 13. ACT 4 — OPERATIONAL LEAKS
# =============================================================================
def chart_16_funnel_by_source(tm, d, tx):
    traffic_by_src = d["web_traffic"].groupby("traffic_source")["sessions"].sum()
    orders_by_src  = tx.groupby("order_source")["order_id"].nunique()
    revenue_by_src = tx.groupby("order_source")["net_revenue"].sum()
    src = pd.DataFrame({"sessions":traffic_by_src, "orders":orders_by_src,
                        "revenue":revenue_by_src}).fillna(0)
    src["cvr"] = src["orders"] / src["sessions"].clip(lower=1) * 100
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"type":"funnel"}, {"type":"bar"}]],
                        subplot_titles=["Aggregate funnel","Conversion rate by source"])
    totals = dict(Sessions=src["sessions"].sum(), Orders=src["orders"].sum())
    fig.add_trace(go.Funnel(y=list(totals.keys()), x=list(totals.values()),
                            marker={"color":[PALETTE["info"], PALETTE["primary"]]}),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=src["cvr"].sort_values(),
                         y=src["cvr"].sort_values().index,
                         orientation="h", marker_color=PALETTE["accent"]),
                  row=1, col=2)
    # Benchmark line
    fig.add_vline(x=BENCHMARKS["sessions_to_order_cvr"]*100,
                  line_dash="dot", line_color=PALETTE["danger"], row=1, col=2,
                  annotation_text="E-com median", annotation_position="top")
    fig.update_layout(title="<b>Traffic → Order Funnel</b>",
                      height=540, showlegend=False)
    export_fig(fig, "16_funnel")
    return fig


def chart_17_inventory_quadrant(d, tx):
    inv = d["inventory"].copy()
    avg_price = tx.groupby("product_id")["unit_price"].mean().rename("avg_price")
    inv = inv.merge(avg_price, on="product_id", how="left")
    inv["lost_sales_est"] = (inv["stockout_days"] *
                             inv["units_sold"] / 30 *
                             inv["avg_price"]).fillna(0)
    latest = inv.sort_values("snapshot_date").groupby("product_id").tail(1)
    fig = px.scatter(latest, x="days_of_supply", y="sell_through_rate",
                     size="lost_sales_est", color="category",
                     size_max=45, log_x=True,
                     hover_data=["product_name","stockout_days","overstock_flag"],
                     title="<b>Inventory Health Quadrant</b> — Latest snapshot")
    fig.add_hline(y=latest["sell_through_rate"].median(), line_dash="dash")
    fig.add_vline(x=30, line_dash="dash",
                  annotation_text="1-month cover", annotation_position="top")
    fig.update_layout(height=600, xaxis_title="Days of supply (log)",
                      yaxis_title="Sell-through rate")
    export_fig(fig, "17_inventory_quadrant")
    return fig


def chart_18_dos_vs_sellthrough(d):
    inv = d["inventory"]
    fig = px.density_contour(inv, x="days_of_supply", y="sell_through_rate",
                             marginal_x="histogram", marginal_y="histogram",
                             title="<b>DoS vs Sell-through</b> — density")
    fig.update_layout(height=580)
    export_fig(fig, "18_dos_sellthrough")
    return fig


def chart_19_returns_sankey(d, tx):
    ret = d["returns"].merge(tx[["order_id","product_id","category","net_revenue"]],
                             on=["order_id","product_id"], how="left")
    ret["refund_bucket"] = pd.cut(ret["refund_amount"],
                                  bins=[0,50000,200000,500000,1e9],
                                  labels=["<50k","50-200k","200-500k",">500k"])
    flows = (ret.groupby(["category","return_reason","refund_bucket"], observed=True)
                .size().reset_index(name="n"))
    labels = (list(flows["category"].unique()) +
              list(flows["return_reason"].unique()) +
              list(flows["refund_bucket"].astype(str).unique()))
    idx = {l: i for i, l in enumerate(labels)}
    source = (flows["category"].map(idx).tolist() +
              flows["return_reason"].map(idx).tolist())
    target = (flows["return_reason"].map(idx).tolist() +
              flows["refund_bucket"].astype(str).map(idx).tolist())
    value  = flows["n"].tolist() * 2
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=18, color=PALETTE["primary"]),
        link=dict(source=source, target=target, value=value,
                  color="rgba(192,57,43,0.35)")))
    fig.update_layout(title="<b>Return flow</b>: Category → Reason → Refund size",
                      height=680)
    export_fig(fig, "19_returns_sankey")
    return fig


def chart_20_shipping_sla(d, tx):
    sh = d["shipments"].merge(tx[["order_id","region"]].drop_duplicates(),
                              on="order_id", how="left")
    sh["lead_days"] = (sh["delivery_date"] - sh["ship_date"]).dt.days
    fig = px.box(sh.dropna(subset=["region","lead_days"]),
                 x="region", y="lead_days", points="suspectedoutliers",
                 color="region",
                 title="<b>Shipping SLA</b> — delivery lead time by region")
    fig.update_layout(height=540, showlegend=False)
    export_fig(fig, "20_shipping_sla")
    return fig


def chart_21_review_vs_return(d, tx):
    rev = (d["reviews"].groupby("product_id")
                        .agg(avg_rating=("rating","mean"),
                             n_reviews=("review_id","count")).reset_index())
    rt = (tx.groupby("product_id")
                .agg(return_rate=("is_returned","mean"),
                     units=("quantity","sum"),
                     category=("category","first")).reset_index())
    m = rev.merge(rt, on="product_id")
    m = m[m["n_reviews"] >= 20]
    fig = px.scatter(m, x="avg_rating", y="return_rate",
                     size="units", color="category", size_max=40,
                     trendline="ols",
                     title="<b>Quality signal</b> — Review rating vs return rate")
    fig.update_layout(height=580,
                      xaxis_title="Avg rating (1-5)", yaxis_title="Return rate")
    export_fig(fig, "21_review_return")
    return fig


# =============================================================================
# 14. ACT 5 — STRATEGIC OUTLOOK
# =============================================================================
def chart_22_forecast_preview(tm):
    s = tm.set_index("date")["revenue"].asfreq("D").interpolate()
    horizon = 180
    hist = s[-365*2:]
    naive = s.shift(365).iloc[-horizon:]
    resid_std = (s.iloc[-365:] - s.shift(365).iloc[-365:]).std()
    upper = naive + 1.96 * resid_std
    lower = naive - 1.96 * resid_std
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                             line=dict(color=PALETTE["muted"])))
    fig.add_trace(go.Scatter(x=naive.index, y=naive.values,
                             name="Seasonal-naïve forecast",
                             line=dict(color=PALETTE["primary"], width=2)))
    fig.add_trace(go.Scatter(x=list(upper.index)+list(lower.index[::-1]),
                             y=list(upper.values)+list(lower.values[::-1]),
                             fill="toself", fillcolor="rgba(31,122,76,0.15)",
                             line=dict(width=0), name="95% band"))
    fig.update_layout(title="<b>Forecast preview</b> — seasonal-naïve baseline",
                      height=540)
    export_fig(fig, "22_forecast_preview")
    return fig


def chart_23_churn_clv_matrix(cust):
    c = cust.copy()
    c["churn_score"] = (c["recency_days"].rank(pct=True)
                        - c["orders"].rank(pct=True)).clip(0, 1)
    c["clv_score"]   = c["revenue"].rank(pct=True)
    def action(r):
        if r["churn_score"] > 0.6 and r["clv_score"] > 0.6:  return "RETAIN — win-back priority"
        if r["churn_score"] > 0.6 and r["clv_score"] < 0.4:  return "LET GO — low ROI"
        if r["churn_score"] < 0.4 and r["clv_score"] > 0.6:  return "NURTURE — VIP"
        return "GROW — upsell"
    c["action"] = c.apply(action, axis=1)
    fig = px.scatter(c.sample(min(8000,len(c)), random_state=RANDOM_SEED),
                     x="churn_score", y="clv_score", color="action",
                     hover_data=["customer_id","orders","revenue"],
                     opacity=0.55,
                     title="<b>Churn × CLV Action Matrix</b>")
    fig.update_layout(height=620,
                      xaxis_title="Churn risk (→ higher)",
                      yaxis_title="CLV percentile")
    export_fig(fig, "23_churn_clv")
    return fig


def chart_24_traffic_revenue_lag(tm):
    a = tm["sessions"].fillna(method="ffill").fillna(0).values
    b = tm["revenue"].fillna(method="ffill").fillna(0).values
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    lags = np.arange(-14, 15)
    vals = [np.corrcoef(a[max(0,-l):len(a)-max(0,l)],
                        b[max(0,l):len(b)-max(0,-l)])[0,1] for l in lags]
    fig = go.Figure(go.Bar(x=lags, y=vals,
                           marker_color=[PALETTE["primary"] if v > 0
                                         else PALETTE["danger"] for v in vals]))
    fig.update_layout(title="<b>Traffic → Revenue lead/lag</b> — cross-correlation",
                      xaxis_title="Lag (days, + = traffic leads revenue)",
                      yaxis_title="Correlation", height=460)
    export_fig(fig, "24_traffic_lag")
    return fig


def chart_25_action_priority(insights_df=None):
    df = insights_df if insights_df is not None else pd.DataFrame([
        {"action":"Win-back email to At-Risk champions",  "impact":9,"effort":2,"owner":"CRM"},
        {"action":"Reduce stockout on top-20 SKUs",       "impact":8,"effort":5,"owner":"Ops"},
        {"action":"Rethink XL sizing (return spike)",     "impact":6,"effort":3,"owner":"Product"},
        {"action":"Kill bottom-quintile promo campaigns", "impact":7,"effort":1,"owner":"Marketing"},
        {"action":"Expand South region paid-search",      "impact":6,"effort":4,"owner":"Growth"},
        {"action":"Bundle low-AOV categories",            "impact":5,"effort":3,"owner":"Merch"},
    ])
    fig = px.scatter(df, x="effort", y="impact", text="action", color="owner",
                     size=[30]*len(df), size_max=30,
                     title="<b>Action Priority Matrix</b> — Impact × Effort")
    fig.update_traces(textposition="top center")
    fig.add_vline(x=df["effort"].median(), line_dash="dash")
    fig.add_hline(y=df["impact"].median(), line_dash="dash")
    fig.update_layout(height=580,
                      xaxis=dict(title="Effort (1-10)", range=[0,10]),
                      yaxis=dict(title="Impact (1-10)", range=[0,10]))
    export_fig(fig, "25_action_priority")
    return fig


# =============================================================================
# 15. ACT 6 — INDUSTRY-GRADE RETAIL METRICS (NEW)
# =============================================================================
def chart_26_sku_pareto(tx):
    """Lorenz curve of SKU revenue + Gini + 80/20 point."""
    rev = tx.groupby("product_id")["net_revenue"].sum().sort_values(ascending=False)
    pct_sku   = np.arange(1, len(rev)+1) / len(rev) * 100
    pct_cum   = rev.cumsum().values / rev.sum() * 100
    gini      = gini_coefficient(rev.values)
    idx80     = int(np.searchsorted(pct_cum, 80))
    pct_at_80 = pct_sku[idx80] if idx80 < len(pct_sku) else 100

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=[
                            f"Lorenz curve — Gini = {gini:.3f}",
                            "Top-15 SKUs by revenue"])
    # Lorenz
    fig.add_trace(go.Scatter(x=pct_sku, y=pct_cum, mode="lines",
                             line=dict(color=PALETTE["primary"], width=3),
                             name="Cumulative revenue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
                             line=dict(dash="dash", color=PALETTE["muted"]),
                             name="Equal distribution"), row=1, col=1)
    fig.add_vline(x=pct_at_80, line_dash="dot",
                  line_color=PALETTE["danger"],
                  annotation_text=f"{pct_at_80:.1f}% SKUs → 80% revenue",
                  row=1, col=1)

    top15 = rev.head(15).reset_index()
    fig.add_trace(go.Bar(x=top15["net_revenue"], y=top15["product_id"].astype(str),
                         orientation="h", marker_color=PALETTE["accent"]),
                  row=1, col=2)
    fig.update_layout(title="<b>SKU Pareto & Gini</b> — concentration check",
                      height=560, showlegend=False)
    fig.update_xaxes(title_text="% of SKUs (ranked by revenue)", row=1, col=1)
    fig.update_yaxes(title_text="% of cumulative revenue", row=1, col=1)
    export_fig(fig, "26_sku_pareto")
    return fig


def chart_27_survival_curves(cust, tx):
    """Kaplan-Meier: time to first order (activation) per channel.

    Per [signup_after_first_order]: 89% of customers have signup_date AFTER
    first_order — physically impossible. We restrict KM to the 11% reliable
    subset (signup < first_order); chart now reflects only that cohort.
    """
    reliable = cust[cust["signup_after_first_order"] == 0].copy()
    activation = reliable[["acquisition_channel","days_to_first_ord"]].dropna()
    activation = activation[activation["days_to_first_ord"] >= 0]
    activation["event"] = 1
    n_reliable = len(activation)
    n_total = len(cust)

    fig = go.Figure()
    for ch, sub in activation.groupby("acquisition_channel"):
        if len(sub) < 50:
            continue
        km = kaplan_meier(sub["days_to_first_ord"].values, sub["event"].values)
        km = km[km["t"] <= 180]
        fig.add_trace(go.Scatter(x=km["t"], y=km["survival"]*100,
                                 mode="lines", name=str(ch),
                                 line=dict(width=2)))
    fig.add_hline(y=50, line_dash="dash", line_color=PALETTE["muted"],
                  annotation_text="Median (50% activated)")
    fig.update_layout(title=f"<b>Kaplan-Meier — time to first order</b> by acquisition "
                            f"channel  ·  reliable subset only "
                            f"({n_reliable:,}/{n_total:,} = {n_reliable/n_total*100:.1f}%)",
                      xaxis_title="Days since signup",
                      yaxis_title="% still NOT activated",
                      height=560)
    export_fig(fig, "27_survival_curves")
    return fig


def chart_28_size_curve(tx):
    """Actual size mix vs industry norm (S=15 / M=35 / L=35 / XL=15)."""
    sizes = ["S","M","L","XL"]
    t = tx[tx["size"].isin(sizes)].copy()
    mix = (t.groupby(["category","size"])["quantity"].sum()
             .groupby(level=0)
             .apply(lambda x: x / x.sum() * 100)
             .unstack(fill_value=0)[sizes])
    norm = pd.Series(BENCHMARKS["size_curve_norm"])
    deviation = (mix - norm).round(1)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55,0.45],
                        subplot_titles=["Actual size mix %",
                                        "Deviation vs industry norm (pp)"])
    # Actual mix stacked
    for sz in sizes:
        fig.add_trace(go.Bar(y=mix.index, x=mix[sz], orientation="h",
                             name=sz, marker_color=None), row=1, col=1)
    # Deviation heatmap
    fig.add_trace(go.Heatmap(z=deviation.values, x=sizes, y=deviation.index,
                             colorscale="RdBu_r", zmid=0,
                             text=deviation.values, texttemplate="%{text}",
                             colorbar=dict(title="pp vs norm")),
                  row=1, col=2)
    fig.update_layout(title="<b>Size Curve Optimization</b> — actual vs industry norm",
                      barmode="stack", height=max(440, 28*len(mix)))
    fig.update_xaxes(title_text="% of units", row=1, col=1)
    export_fig(fig, "28_size_curve")
    return fig


def chart_29_lifecycle_scurve(tx, top_categories=6, horizon_weeks=26):
    """Launch-normalized lifecycle S-curves per category."""
    launch = tx.groupby("product_id")["order_date"].min().rename("launch")
    t = tx.merge(launch.reset_index(), on="product_id")
    t["week"] = ((t["order_date"] - t["launch"]).dt.days // 7)
    t = t[(t["week"] >= 0) & (t["week"] <= horizon_weeks)]

    weekly = t.groupby(["product_id","category","week"])["quantity"].sum().reset_index()
    weekly["norm"] = weekly.groupby("product_id")["quantity"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0)
    # Keep only top N categories by revenue
    top_cat = (tx.groupby("category")["net_revenue"].sum()
                 .nlargest(top_categories).index)
    weekly = weekly[weekly["category"].isin(top_cat)]

    avg = weekly.groupby(["category","week"])["norm"].mean().reset_index()
    fig = px.line(avg, x="week", y="norm", color="category", markers=False,
                  title="<b>Product Lifecycle S-Curves</b> — "
                        "launch-normalized weekly demand")
    fig.add_vline(x=BENCHMARKS["product_lifecycle_weeks_fast_fashion"],
                  line_dash="dot", line_color=PALETTE["accent"],
                  annotation_text="Fast-fashion peak")
    fig.add_vline(x=BENCHMARKS["product_lifecycle_weeks_traditional"],
                  line_dash="dot", line_color=PALETTE["info"],
                  annotation_text="Traditional peak")
    fig.update_layout(height=540,
                      xaxis_title="Weeks since launch",
                      yaxis_title="Normalized demand (% of peak)")
    export_fig(fig, "29_lifecycle_scurve")
    return fig


def chart_30_gmroi(tx, d):
    """GMROI = Gross Profit / Avg Inventory@Cost, per category."""
    margin_by_cat = tx.groupby("category")["gross_profit"].sum()
    # inventory already has 'category'; only merge cogs to value the stock
    prod_cogs = tx[["product_id","cogs"]].drop_duplicates("product_id")
    inv = d["inventory"].merge(prod_cogs, on="product_id", how="left",
                               suffixes=("", "_prod"))
    cogs_col = "cogs" if "cogs" in inv.columns else "cogs_prod"
    inv["inv_value"] = inv["stock_on_hand"] * inv[cogs_col]
    avg_inv_by_cat = inv.groupby("category")["inv_value"].mean()
    gmroi = (margin_by_cat / avg_inv_by_cat).dropna().sort_values()
    fig = go.Figure(go.Bar(x=gmroi.values, y=gmroi.index, orientation="h",
                           marker=dict(color=gmroi.values,
                                       colorscale="RdYlGn",
                                       cmid=BENCHMARKS["gmroi_target"]),
                           text=gmroi.round(2).values, textposition="outside"))
    fig.add_vline(x=BENCHMARKS["gmroi_target"], line_dash="dash",
                  line_color=PALETTE["danger"],
                  annotation_text=f"Target GMROI ≥ {BENCHMARKS['gmroi_target']}")
    fig.update_layout(title="<b>GMROI per Category</b> — "
                            "gross profit on each $ of inventory",
                      xaxis_title="GMROI (higher = better)",
                      height=max(440, 28*len(gmroi)))
    export_fig(fig, "30_gmroi")
    return fig


def chart_31_promo_incrementality(tx):
    """Avg daily revenue per SKU on promo-day vs non-promo-day (DiD proxy)."""
    t = tx.copy()
    t["date"] = t["order_date"].dt.date
    daily = (t.groupby(["product_id","category","date","is_promo"])["net_revenue"]
              .sum().reset_index())
    uplift = (daily.groupby(["product_id","category","is_promo"])["net_revenue"]
                   .mean().unstack(fill_value=np.nan))
    uplift.columns = ["baseline" if c == 0 else "promo" for c in uplift.columns]
    uplift = uplift.dropna()
    uplift["lift_pct"] = (uplift["promo"] - uplift["baseline"]) / uplift["baseline"] * 100
    cat_lift = (uplift.reset_index()
                      .groupby("category")["lift_pct"].median()
                      .sort_values())
    colors = ["#1F7A4C" if v > 30 else
              "#E8A33D" if v > 0 else
              "#C0392B" for v in cat_lift.values]
    fig = go.Figure(go.Bar(x=cat_lift.values, y=cat_lift.index,
                           orientation="h", marker_color=colors,
                           text=cat_lift.round(1), textposition="outside"))
    fig.add_vline(x=0, line_color=PALETTE["muted"])
    fig.add_vline(x=30, line_dash="dash", line_color=PALETTE["muted"],
                  annotation_text="30% lift threshold")
    fig.update_layout(title="<b>Promo Incrementality</b> — median per-SKU lift % "
                            "(promo day vs non-promo day)",
                      xaxis_title="Median revenue lift per SKU (%)",
                      height=max(440, 28*len(cat_lift)))
    export_fig(fig, "31_promo_incrementality")
    return fig


def chart_32_anomaly_stl(tm, promos):
    """STL-residual |z|>3 anomalies with promo overlay."""
    s = tm.set_index("date")["revenue"].asfreq("D").interpolate()
    stl = STL(s, period=365, robust=True).fit()
    resid = stl.resid
    z = (resid - resid.mean()) / resid.std()
    anomalies_idx = z.index[z.abs() > 3]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Daily revenue with anomalies",
                                        "STL residual z-score"],
                        vertical_spacing=0.08, row_heights=[0.6,0.4])
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                             line=dict(color=PALETTE["primary"], width=1),
                             name="Revenue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=anomalies_idx, y=s.loc[anomalies_idx].values,
                             mode="markers",
                             marker=dict(color=PALETTE["danger"], size=8,
                                         symbol="x"),
                             name="Anomaly (|z|>3)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=z.index, y=z.values, mode="lines",
                             line=dict(color=PALETTE["muted"], width=1),
                             showlegend=False), row=2, col=1)
    for y in (3, -3):
        fig.add_hline(y=y, line_dash="dash", line_color=PALETTE["danger"],
                      row=2, col=1)
    # Promo overlays
    top = (promos.assign(span=(promos["end_date"]-promos["start_date"]).dt.days)
                  .sort_values("span", ascending=False).head(6))
    for _, p in top.iterrows():
        fig.add_vrect(x0=p["start_date"], x1=p["end_date"],
                      fillcolor=PALETTE["accent"], opacity=0.10,
                      line_width=0, row=1, col=1)

    anomaly_pct = len(anomalies_idx) / len(z) * 100
    fig.update_layout(title=f"<b>Anomaly Detection</b> — "
                            f"{len(anomalies_idx)} anomaly days "
                            f"({anomaly_pct:.1f}% of series)",
                      height=640)
    export_fig(fig, "32_anomaly_stl")
    return fig


# =============================================================================
# 16. ACT 7 — UNIT ECONOMICS (v3 strategist feedback)
# =============================================================================
def _estimate_variable_costs(tx, d):
    """Variable costs (shipping, payment, packaging, returns) + scenarios.

    The shipping line in the dataset is the FEE COLLECTED FROM CUSTOMERS,
    not the COST PAID TO CARRIERS. Industry benchmark for actual cost is
    8-12% of revenue (Reichheld & Schefter HBR 2000; Narvar 2023). The
    'as_reported' value is a floor; 'industry_low' / 'industry_high' are
    realistic stress-tested scenarios."""
    net_rev = tx["net_revenue"].sum()
    shipping_reported = d["shipments"]["shipping_fee"].fillna(0).sum()

    rates = BENCHMARKS["payment_fee_rates"]
    pay = d["payments"].copy()
    pay["rate"] = pay["payment_method"].map(rates).fillna(0.025)
    payment_fee = (pay["payment_value"].fillna(0) * pay["rate"]).sum()

    packaging = net_rev * BENCHMARKS["packaging_cost_pct"]
    return_proc = net_rev * BENCHMARKS["return_processing_pct"]
    refunds_out = tx["refund_amt"].sum()

    # Shipping scenarios for sensitivity analysis
    shipping_scenarios = dict(
        as_reported   = shipping_reported,
        industry_low  = net_rev * 0.08,
        industry_high = net_rev * 0.12,
    )
    return dict(shipping=shipping_reported,
                shipping_scenarios=shipping_scenarios,
                payment=payment_fee,
                packaging=packaging,
                return_proc=return_proc,
                refunds=refunds_out)


def chart_33_contribution_margin(tx, d):
    """v6: Single waterfall — goods-only CM per [A9_shipping_treatment].

    Shipping EXCLUDED. Carrier cost not in dataset; customer-paid fee shown
    only as informational memo annotation (not in CM total).

    Per [A3_realized_order_def]: filter cancelled to match Section 1 KPI
    metric (was a bug — chart used unfiltered tx and reported +2.79%; metric
    used tx_real and reported +2.26%).
    """
    tx_real = tx[tx["is_cancelled"] == 0]                  # match compute_metrics scope
    vc = _estimate_variable_costs(tx_real, d)
    net_rev = tx_real["net_revenue"].sum()
    cogs    = tx_real["cogs_total"].sum()
    gross_profit = net_rev - cogs

    non_ship_vc = vc["payment"] + vc["packaging"] + vc["return_proc"] + vc["refunds"]
    cm = gross_profit - non_ship_vc       # goods-only per A9
    cm_pct = cm / net_rev * 100

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","total",
                 "relative","relative","relative","relative","total"],
        x=["Net Revenue","−COGS","Gross Profit",
           "−Payment fees","−Packaging","−Return proc","−Refunds",
           "CM (goods-only)"],
        y=[net_rev, -cogs, gross_profit,
           -vc["payment"], -vc["packaging"], -vc["return_proc"], -vc["refunds"],
           cm],
        connector={"line":{"color":PALETTE["muted"]}},
        decreasing={"marker":{"color":PALETTE["danger"]}},
        increasing={"marker":{"color":PALETTE["primary"]}},
        totals    ={"marker":{"color":PALETTE["accent"]}},
        text=[f"{v:,.0f}" for v in
              [net_rev,-cogs,gross_profit,
               -vc["payment"], -vc["packaging"],
               -vc["return_proc"], -vc["refunds"], cm]],
        textposition="outside"))

    # Memo annotation for shipping (NOT in CM)
    ship_pct_rev = vc["shipping"] / max(net_rev, 1) * 100
    fig.add_annotation(
        x=0.5, y=1.06, xref="paper", yref="paper",
        text=(f"<i>Memo: shipping fee from customers = "
              f"{vc['shipping']:,.0f} VND ({ship_pct_rev:.3f}% of revenue) — "
              f"NOT in CM per [A9_shipping_treatment]; carrier cost not in dataset.</i>"),
        showarrow=False, font=dict(size=11, color=PALETTE["muted"]))

    fig.update_layout(
        title=f"<b>Contribution Margin Waterfall (goods-only)</b>  ·  "
              f"CM = {cm_pct:+.2f}% of net revenue  ·  "
              f"shipping excluded per A9",
        yaxis_title="VND (scaled)", height=560,
        margin=dict(t=110))
    export_fig(fig, "33_contribution_margin")
    return fig


def chart_34_revenue_decomposition(tm, tx):
    """Revenue = Traffic × CVR × UPT × AUR, indexed to month 1 = 100."""
    # Monthly aggregation
    tm2 = tm.copy()
    tm2["ym"] = tm2["date"].dt.to_period("M").dt.to_timestamp()
    agg = tm2.groupby("ym").agg(
        sessions=("sessions","sum"),
        orders=("orders","sum"),
        revenue=("revenue","sum"),
        units=("units","sum")).reset_index()
    # Guard against zeros
    agg["CVR"] = agg["orders"] / agg["sessions"].clip(lower=1)
    agg["UPT"] = agg["units"] / agg["orders"].clip(lower=1)
    agg["AUR"] = agg["revenue"] / agg["units"].clip(lower=1)
    agg["Traffic"] = agg["sessions"]

    # Index from first valid month
    base = agg.iloc[0]
    for col in ["Traffic","CVR","UPT","AUR"]:
        agg[f"{col}_idx"] = agg[col] / base[col] * 100

    fig = go.Figure()
    colors = {"Traffic":PALETTE["info"], "CVR":PALETTE["primary"],
              "UPT":PALETTE["accent"], "AUR":PALETTE["danger"]}
    for driver in ["Traffic","CVR","UPT","AUR"]:
        fig.add_trace(go.Scatter(x=agg["ym"], y=agg[f"{driver}_idx"],
                                 mode="lines", name=driver,
                                 line=dict(color=colors[driver], width=2)))
    fig.add_hline(y=100, line_dash="dash", line_color=PALETTE["muted"],
                  annotation_text="Baseline (month 1 = 100)")
    fig.update_layout(
        title="<b>Revenue Decomposition</b> — Traffic × CVR × UPT × AUR (indexed)",
        xaxis_title="Month", yaxis_title="Index (base = 100)", height=540)
    export_fig(fig, "34_revenue_decomposition")
    return fig


# =============================================================================
# 17. ACT 8 — VIETNAM MARKET CONTEXT (v3)
# =============================================================================
def chart_35_cod_cancellation(d, tx):
    orders = d["orders"].merge(d["geography"][["zip","region"]], on="zip", how="left")
    orders["is_cancelled"] = (orders["order_status"] == "cancelled").astype(int)

    # Cancel rate by payment_method
    by_pm = (orders.groupby("payment_method")["is_cancelled"].mean() * 100).sort_values()

    # Payment × Region heatmap
    piv = (orders.groupby(["payment_method","region"])["is_cancelled"].mean()
                   .unstack() * 100)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=["Cancel rate by payment method",
                                        "Payment × Region (% cancelled)"])
    colors = [PALETTE["danger"] if v > BENCHMARKS["cod_cancel_rate_vn"]*100
              else PALETTE["primary"] for v in by_pm.values]
    fig.add_trace(go.Bar(x=by_pm.values, y=by_pm.index, orientation="h",
                         marker_color=colors, text=by_pm.round(1),
                         textposition="outside"), row=1, col=1)
    fig.add_vline(x=BENCHMARKS["cod_cancel_rate_vn"]*100, line_dash="dot",
                  line_color=PALETTE["danger"],
                  annotation_text=f"VN COD benchmark "
                                  f"{BENCHMARKS['cod_cancel_rate_vn']:.0%}",
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                             colorscale="Reds",
                             text=piv.round(1).values, texttemplate="%{text}%",
                             colorbar=dict(title="% cancel")),
                  row=1, col=2)
    fig.update_layout(title="<b>COD Cancellation Diagnostic</b> — "
                            "đơn không nhận đốt logistics cost",
                      showlegend=False, height=520)
    export_fig(fig, "35_cod_cancellation")
    return fig


def _vn_event_windows(year_min, year_max):
    """Generate (name, start_date, end_date, year) tuples for every VN event."""
    out = []
    for y in range(year_min, year_max + 1):
        # Tết (7-day window centered on D-day, +3 after)
        if y in TET_DATES:
            d0 = pd.Timestamp(TET_DATES[y])
            out.append(("Tết", d0 - pd.Timedelta(3, "d"),
                        d0 + pd.Timedelta(3, "d"), y))
        for name, md, span in VN_FIXED_EVENTS:
            try:
                d0 = pd.Timestamp(f"{y}-{md}")
            except Exception:
                continue
            out.append((name, d0, d0 + pd.Timedelta(span-1, "d"), y))
    return out


def chart_36_vn_calendar_lift(tm):
    years = sorted(tm["year"].dropna().unique().astype(int))
    events = _vn_event_windows(min(years), max(years))

    lifts = []
    for name, start, end, year in events:
        during = tm[(tm.date >= start) & (tm.date <= end)]["revenue"].mean()
        before = tm[(tm.date >= start - pd.Timedelta(30,"d")) &
                    (tm.date <  start)]["revenue"].mean()
        after  = tm[(tm.date >  end) &
                    (tm.date <= end + pd.Timedelta(30,"d"))]["revenue"].mean()
        base = np.nanmean([before, after])
        if np.isnan(during) or np.isnan(base) or base <= 0:
            continue
        lifts.append(dict(event=name, year=year,
                          lift_pct=(during - base) / base * 100))
    df = pd.DataFrame(lifts)
    if df.empty:
        df = pd.DataFrame([{"event":"(no events matched)", "year":0, "lift_pct":0}])

    median_lift = df.groupby("event")["lift_pct"].median().sort_values()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=["Per-year lift % per event",
                                        "Median lift % by event"])
    for ev in df["event"].unique():
        sub = df[df["event"] == ev]
        fig.add_trace(go.Box(x=[ev]*len(sub), y=sub["lift_pct"],
                             name=ev, boxmean=True, showlegend=False),
                      row=1, col=1)
    colors = [PALETTE["primary"] if v > 0 else PALETTE["danger"]
              for v in median_lift.values]
    fig.add_trace(go.Bar(x=median_lift.values, y=median_lift.index,
                         orientation="h", marker_color=colors,
                         text=median_lift.round(0), textposition="outside"),
                  row=1, col=2)
    fig.add_vline(x=0, line_color=PALETTE["muted"], row=1, col=2)
    fig.update_layout(title="<b>Vietnam Calendar Event Lift</b> — "
                            "revenue uplift % vs ±30-day baseline",
                      height=600)
    export_fig(fig, "36_vn_calendar_lift")
    return fig


# =============================================================================
# 18. ACT 9 — FASHION MERCHANDISING (v3)
# =============================================================================
def chart_37_str_week6(tx, d):
    launch = tx.groupby("product_id")["order_date"].min().rename("launch")
    t = tx.merge(launch.reset_index(), on="product_id")
    t["days_since_launch"] = (t["order_date"] - t["launch"]).dt.days

    # Full-price units sold in first 42 days per SKU
    fp_6w = (t[(t["days_since_launch"] <= 42) & (t["is_full_price"] == 1)]
             .groupby("product_id")["quantity"].sum().rename("fp_units_6w"))

    # Total lifetime units received per SKU (from inventory snapshots)
    received = d["inventory"].groupby("product_id")["units_received"].sum().rename("received")

    cat = tx[["product_id","category"]].drop_duplicates("product_id")
    m = (fp_6w.to_frame().join(received, how="inner")
                        .reset_index()
                        .merge(cat, on="product_id", how="left"))
    m = m[m["received"] > 0]
    m["str_6w_pct"] = (m["fp_units_6w"] / m["received"] * 100).clip(upper=100)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=[
                            f"Distribution of Week-6 Full-Price STR "
                            f"(n={len(m):,} SKUs)",
                            "By category (median ± IQR)"])
    fig.add_trace(go.Histogram(x=m["str_6w_pct"], nbinsx=40,
                               marker_color=PALETTE["primary"]), row=1, col=1)
    fig.add_vline(x=BENCHMARKS["full_price_str_week6_healthy"]*100,
                  line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Healthy ≥ "
                                  f"{BENCHMARKS['full_price_str_week6_healthy']:.0%}",
                  row=1, col=1)
    # Category box
    top_cats = m["category"].value_counts().head(10).index
    for c in top_cats:
        sub = m[m["category"] == c]
        fig.add_trace(go.Box(y=sub["str_6w_pct"], name=c,
                             showlegend=False), row=1, col=2)
    pct_healthy = (m["str_6w_pct"] >=
                   BENCHMARKS["full_price_str_week6_healthy"]*100).mean()
    fig.update_layout(title=f"<b>Full-Price Sell-Through @ Week 6</b> — "
                            f"{pct_healthy:.1%} SKUs pass healthy threshold",
                      height=560)
    fig.update_xaxes(title_text="STR % in first 42 days", row=1, col=1)
    fig.update_yaxes(title_text="STR @ Week 6 (%)", row=1, col=2)
    export_fig(fig, "37_str_week6")
    return fig


def chart_38_aged_newness(tx, d):
    # Aged inventory at latest snapshot
    launch = tx.groupby("product_id")["order_date"].min().rename("launch")
    latest = (d["inventory"].sort_values("snapshot_date")
                             .groupby("product_id").tail(1)
                             .merge(launch.reset_index(), on="product_id", how="left"))
    # Merge cogs to value stock
    cogs = tx[["product_id","cogs"]].drop_duplicates("product_id")
    latest = latest.merge(cogs, on="product_id", how="left", suffixes=("","_c"))
    cogs_col = "cogs_c" if "cogs_c" in latest.columns else "cogs"
    latest["inv_value"] = latest["stock_on_hand"] * latest[cogs_col]
    latest["age_days"] = (latest["snapshot_date"] - latest["launch"]).dt.days
    bins   = [-1, 30, 60, 90, 180, 10000]
    labels = ["0-30","30-60","60-90","90-180","180+"]
    latest["age_bucket"] = pd.cut(latest["age_days"], bins=bins, labels=labels)
    bucket_val = (latest.groupby("age_bucket", observed=True)["inv_value"]
                         .sum().reindex(labels).fillna(0))

    # Newness ratio per quarter
    t = tx.merge(launch.reset_index(), on="product_id")
    t["age_at_order"] = (t["order_date"] - t["launch"]).dt.days
    t["is_new"] = (t["age_at_order"] < 90).astype(int)
    t["quarter"] = t["order_date"].dt.to_period("Q").dt.to_timestamp()
    q_agg = t.groupby("quarter").apply(
        lambda x: x.loc[x.is_new == 1, "net_revenue"].sum()
                  / max(x["net_revenue"].sum(), 1) * 100).rename("newness_pct").reset_index()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=["Inventory value by age bucket",
                                        "Newness ratio (% revenue <90d old)"])
    bar_colors = ["#1F7A4C","#52BE80","#F5CBA7","#E8A33D","#C0392B"]
    fig.add_trace(go.Bar(x=labels, y=bucket_val.values,
                         marker_color=bar_colors,
                         text=[f"{v:,.0f}" for v in bucket_val.values],
                         textposition="outside"), row=1, col=1)
    fig.add_trace(go.Scatter(x=q_agg["quarter"], y=q_agg["newness_pct"],
                             mode="lines+markers",
                             line=dict(color=PALETTE["primary"], width=2)),
                  row=1, col=2)
    fig.add_hline(y=BENCHMARKS["newness_ratio_zara_shein"]*100,
                  line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Zara/Shein benchmark "
                                  f"{BENCHMARKS['newness_ratio_zara_shein']:.0%}",
                  row=1, col=2)
    fig.update_layout(title="<b>Aged Inventory + Newness Ratio</b>",
                      showlegend=False, height=540)
    fig.update_xaxes(title_text="Age bucket (days)", row=1, col=1)
    fig.update_yaxes(title_text="Inventory value (VND at cost)", row=1, col=1)
    fig.update_yaxes(title_text="Newness %", row=1, col=2)
    export_fig(fig, "38_aged_newness")
    return fig


def chart_41_price_paradox(tx, d=None):
    """v6 upgrade: 3-layer margin compression diagnostic.

    Reveals 2 distinct margin-loss layers:
      (1) Selection bias: full catalog vs sold-only SKUs — high-margin SKUs
          systematically unsold (catalogue padding for v6 dataset).
      (2) Discount erosion: list price → unit_price after promo.
    """
    # Layer 1: Full catalog list margin (source-of-truth)
    if d is not None:
        prods_full = d["products"][["product_id","category","price","cogs"]].copy()
    else:
        # Fallback if d not passed — derive from tx (covers sold SKUs only)
        prods_full = tx[["product_id","category","price","cogs"]].drop_duplicates("product_id")
    prods_full["list_margin"] = (prods_full["price"] - prods_full["cogs"]) / prods_full["price"] * 100

    # Layer 2: Sold-only SKUs list margin
    sold_skus = set(tx["product_id"].unique())
    p_sold = prods_full[prods_full["product_id"].isin(sold_skus)].copy()
    p_unsold = prods_full[~prods_full["product_id"].isin(sold_skus)].copy()

    # Layer 3: Realized margin per SKU (avg unit_price after promo)
    realized = (tx.groupby("product_id")
                  .agg(avg_unit_price=("unit_price","mean"),
                       cogs=("cogs","first"),
                       category=("category","first")).reset_index())
    realized["realized_margin"] = ((realized["avg_unit_price"] - realized["cogs"])
                                    / realized["avg_unit_price"] * 100)

    med_full     = float(prods_full["list_margin"].median())
    med_sold     = float(p_sold["list_margin"].median())
    med_unsold   = float(p_unsold["list_margin"].median()) if len(p_unsold) else float("nan")
    med_realized = float(realized["realized_margin"].median())

    selection_bias    = med_full - med_sold     # 30.6 - 19.8 = 10.8pp
    discount_erosion  = med_sold - med_realized  # 19.8 - 14.6 = 5.2pp

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=[
                            "Margin compression — 3 layers (catalog → sold → realized)",
                            "Price vs COGS — sold SKUs (color = list margin %)"])

    # Panel 1: 3-layer waterfall
    fig.add_trace(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","total"],
        x=["List margin<br>(full catalog)",
           "Selection bias<br>(unsold SKUs lost)",
           "Discount erosion<br>(promo)",
           "Realized margin<br>(after promo)"],
        y=[med_full, -selection_bias, -discount_erosion, med_realized],
        connector={"line":{"color":PALETTE["muted"]}},
        decreasing={"marker":{"color":PALETTE["danger"]}},
        increasing={"marker":{"color":PALETTE["primary"]}},
        totals    ={"marker":{"color":PALETTE["accent"]}},
        text=[f"{med_full:.1f}%", f"−{selection_bias:.1f}pp",
              f"−{discount_erosion:.1f}pp", f"{med_realized:.1f}%"],
        textposition="outside"), row=1, col=1)

    # Panel 2: Price vs COGS scatter (sold SKUs only)
    fig.add_trace(go.Scatter(x=p_sold["price"], y=p_sold["cogs"], mode="markers",
                             marker=dict(size=6, opacity=0.35,
                                         color=p_sold["list_margin"],
                                         colorscale="RdYlGn",
                                         colorbar=dict(title="Margin %", x=1.02)),
                             text=p_sold["category"], showlegend=False),
                  row=1, col=2)
    mx = max(p_sold["price"].max(), p_sold["cogs"].max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(dash="dash", color=PALETTE["danger"]),
                             name="y=x (margin=0)", showlegend=False), row=1, col=2)

    pct_negative_full = (prods_full["list_margin"] < 0).mean()
    fig.update_layout(
        title=(f"<b>Price–COGS Paradox</b> — Margin compression: "
               f"catalog {med_full:.1f}% → sold-only {med_sold:.1f}% "
               f"(−{selection_bias:.1f}pp selection) → realized {med_realized:.1f}% "
               f"(−{discount_erosion:.1f}pp promo)  ·  "
               f"unsold catalog padding: {len(p_unsold)} SKUs"),
        height=600)
    fig.update_yaxes(title_text="Margin %", row=1, col=1)
    fig.update_xaxes(title_text="Price (scaled VND)", row=1, col=2)
    fig.update_yaxes(title_text="COGS (scaled VND)", row=1, col=2)
    export_fig(fig, "41_price_paradox")
    return fig


# =============================================================================
# 19. ACT 10 — CUSTOMER BEHAVIOR DEEP-DIVE (v3)
# =============================================================================
def chart_39_category_penetration(cust, tx):
    # Co-occurrence matrix via one-hot pivot
    flag = pd.crosstab(tx["customer_id"], tx["category"]) > 0
    cooc = flag.T.astype(int) @ flag.astype(int)
    diag = np.diag(cooc.values).astype(float)
    penetration = cooc.div(diag, axis=0) * 100  # P(B|A) per row A

    # First-purchase category
    first_cat = (tx.sort_values("order_date")
                   .groupby("customer_id")["category"].first()
                   .rename("first_category").reset_index())
    cust_fc = cust.merge(first_cat, on="customer_id", how="left")
    ltv_by_fc = (cust_fc.groupby("first_category")["revenue"]
                         .agg(["median","mean","count"])
                         .sort_values("median"))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=[
                            "P(buy B | bought A) × 100",
                            "Median CLV by FIRST-purchase category"])
    fig.add_trace(go.Heatmap(z=penetration.values,
                             x=penetration.columns, y=penetration.index,
                             colorscale="Greens",
                             text=np.round(penetration.values, 0),
                             texttemplate="%{text}",
                             colorbar=dict(title="%")), row=1, col=1)
    fig.add_trace(go.Bar(x=ltv_by_fc["median"], y=ltv_by_fc.index,
                         orientation="h", marker_color=PALETTE["primary"],
                         text=ltv_by_fc["median"].round(0),
                         textposition="outside"), row=1, col=2)
    fig.update_layout(title="<b>Category Penetration & First-Purchase → LTV</b>",
                      height=max(520, 28*len(penetration)),
                      showlegend=False)
    fig.update_xaxes(title_text="Bought also (B)", row=1, col=1)
    fig.update_yaxes(title_text="Starting from (A)", row=1, col=1)
    fig.update_xaxes(title_text="Median CLV (VND)", row=1, col=2)
    export_fig(fig, "39_category_penetration")
    return fig


def chart_40_wardrobing(tx, d):
    # Per-customer return metrics
    cr = (tx.groupby("customer_id")
            .agg(return_rate=("is_returned","mean"),
                 orders=("order_id","nunique"),
                 revenue=("net_revenue","sum"),
                 refunds=("refund_amt","sum")).reset_index())
    cr["wardrobing_flag"] = ((cr["return_rate"] > BENCHMARKS["wardrobing_return_rate"]) &
                              (cr["orders"] >= 3)).astype(int)

    # Cancel vs Return split
    orders = d["orders"]
    n_cancel = (orders["order_status"] == "cancelled").sum()
    n_return = tx.loc[tx["is_returned"] == 1, "order_id"].nunique()
    n_delivered = (orders["order_status"].isin(["delivered","returned"])).sum()
    n_ok = n_delivered - n_return

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        specs=[[{"type":"xy"}, {"type":"domain"}]],
                        subplot_titles=[
                            "Customer return-rate distribution "
                            f"(wardrobers = {cr['wardrobing_flag'].sum():,} "
                            f"or {cr['wardrobing_flag'].mean():.2%})",
                            "Order outcome split"])
    fig.add_trace(go.Histogram(x=cr["return_rate"]*100, nbinsx=50,
                               marker_color=PALETTE["primary"]), row=1, col=1)
    fig.add_vline(x=BENCHMARKS["wardrobing_return_rate"]*100,
                  line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Wardrobing threshold "
                                  f"{BENCHMARKS['wardrobing_return_rate']:.0%}",
                  row=1, col=1)
    fig.add_trace(go.Pie(labels=["Delivered OK","Returned post-delivery",
                                 "Pre-delivery cancelled"],
                         values=[max(n_ok,0), n_return, n_cancel],
                         marker=dict(colors=[PALETTE["primary"],
                                             PALETTE["accent"],
                                             PALETTE["danger"]]),
                         hole=0.4), row=1, col=2)
    fig.update_layout(title="<b>Wardrobing Flag + Order Outcome Split</b>",
                      height=520, showlegend=True)
    fig.update_xaxes(title_text="Customer return rate %", row=1, col=1)
    export_fig(fig, "40_wardrobing")
    return fig


# =============================================================================
# 20. ACT 11 — TEXT MINING & ESG (v3)
# =============================================================================
# Minimal VN+EN stopwords — enough for first-pass keyword extraction
_STOP = set("""
a an the and or but so of to in on at for with from by is are was were be been being
this that these those it its as not no yes i me my we our you your he she they them
tôi mình bạn anh chị em là rất và với của cho tại khi vì nên nếu đã đang sẽ không có
một hai ba bốn năm ở rồi lại được theo cũng vẫn đây đó cái chiếc
the rất tốt đẹp tuyệt vời ok oki nice good bad ổn ok
""".split())


def _tokenize(text):
    import re
    text = str(text).lower()
    toks = re.findall(r"[a-zA-ZÀ-ỹà-ỹ]+", text)
    return [t for t in toks if t not in _STOP and len(t) >= 3]


def chart_42_review_nps(d):
    rv = d["reviews"].copy()
    rv["rating_bucket"] = pd.cut(rv["rating"], bins=[0,2,3,5],
                                  labels=["Negative (1-2★)",
                                          "Neutral (3★)",
                                          "Positive (4-5★)"])

    # NPS proxy (standard: 5=promoter, 4=passive, 1-3=detractor)
    promoters  = (rv["rating"] == 5).sum()
    detractors = (rv["rating"] <= 3).sum()
    total      = len(rv)
    nps = (promoters - detractors) / total * 100 if total else 0

    # Keyword frequencies per bucket (top 15)
    rv["tokens"] = rv["review_title"].apply(_tokenize)
    kw = {}
    for b in rv["rating_bucket"].cat.categories:
        all_tokens = []
        for toks in rv.loc[rv["rating_bucket"] == b, "tokens"]:
            all_tokens.extend(toks)
        kw[b] = Counter(all_tokens).most_common(15)

    fig = make_subplots(rows=1, cols=4,
                        specs=[[{"type":"indicator"},
                                {"type":"xy"},{"type":"xy"},{"type":"xy"}]],
                        column_widths=[0.18, 0.27, 0.27, 0.28],
                        subplot_titles=["NPS proxy",
                                        "Top words — Negative",
                                        "Top words — Neutral",
                                        "Top words — Positive"])
    fig.add_trace(go.Indicator(mode="number+gauge", value=nps,
                               number={"valueformat":".0f"},
                               gauge={"axis":{"range":[-100,100]},
                                      "bar":{"color":PALETTE["primary"]},
                                      "steps":[
                                          {"range":[-100,0],"color":"#F8D7DA"},
                                          {"range":[0,30],"color":"#FFF3CD"},
                                          {"range":[30,100],"color":"#D4EDDA"}]}),
                  row=1, col=1)
    colors = [PALETTE["danger"], PALETTE["muted"], PALETTE["primary"]]
    for i, (b, color) in enumerate(zip(kw.keys(), colors), start=2):
        words = kw[b]
        if not words:
            continue
        df_kw = pd.DataFrame(words, columns=["word","count"])
        fig.add_trace(go.Bar(x=df_kw["count"][::-1], y=df_kw["word"][::-1],
                             orientation="h", marker_color=color,
                             showlegend=False), row=1, col=i)
    fig.update_layout(title=f"<b>Review Text + NPS Proxy</b> — "
                            f"NPS ≈ {nps:.0f} · {total:,} reviews",
                      height=560)
    export_fig(fig, "42_review_nps")
    return fig


def chart_43_esg_risks(tx, d):
    # (a) Return carbon footprint
    n_returns = d["returns"]["return_quantity"].sum()
    co2_kg = n_returns * BENCHMARKS["carbon_per_return_kg"]
    co2_tonnes = co2_kg / 1000

    # (b) Deadstock risk SKUs
    launch = tx.groupby("product_id")["order_date"].min().rename("launch")
    latest = (d["inventory"].sort_values("snapshot_date")
                             .groupby("product_id").tail(1)
                             .merge(launch.reset_index(), on="product_id", how="left"))
    cogs = tx[["product_id","cogs"]].drop_duplicates("product_id")
    latest = latest.merge(cogs, on="product_id", how="left", suffixes=("","_c"))
    ccol = "cogs_c" if "cogs_c" in latest.columns else "cogs"
    latest["inv_value"] = latest["stock_on_hand"] * latest[ccol]
    latest["age_days"]  = (latest["snapshot_date"] - latest["launch"]).dt.days
    latest["is_deadstock"] = ((latest["age_days"] > BENCHMARKS["deadstock_days"]) &
                               (latest["sell_through_rate"] < 0.3)).astype(int)
    deadstock_val = latest.loc[latest["is_deadstock"] == 1, "inv_value"].sum()
    total_inv_val = latest["inv_value"].sum()
    deadstock_pct = deadstock_val / max(total_inv_val, 1) * 100

    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type":"indicator"},{"type":"indicator"},
                                {"type":"xy"}]],
                        column_widths=[0.22, 0.22, 0.56],
                        subplot_titles=["Returns CO₂ estimate",
                                        "Deadstock % of inv value",
                                        "Top-20 deadstock SKUs by value"])
    fig.add_trace(go.Indicator(mode="number+delta",
                               value=co2_tonnes,
                               number={"suffix":" t CO₂",
                                       "font":{"color":PALETTE["danger"]}},
                               title={"text":"Est. from returns<br>(0.5 kg/return)"}),
                  row=1, col=1)
    fig.add_trace(go.Indicator(mode="number+gauge",
                               value=deadstock_pct,
                               number={"suffix":"%",
                                       "font":{"color":PALETTE["danger"]}},
                               gauge={"axis":{"range":[0,50]},
                                      "bar":{"color":PALETTE["danger"]},
                                      "steps":[
                                          {"range":[0,5],"color":"#D4EDDA"},
                                          {"range":[5,10],"color":"#FFF3CD"},
                                          {"range":[10,50],"color":"#F8D7DA"}]}),
                  row=1, col=2)
    top = (latest[latest["is_deadstock"] == 1]
           .nlargest(20, "inv_value"))
    if not top.empty:
        fig.add_trace(go.Bar(x=top["inv_value"],
                             y=top["product_id"].astype(str),
                             orientation="h",
                             marker_color=PALETTE["danger"],
                             text=top["age_days"].astype(int),
                             textposition="outside"),
                      row=1, col=3)
    fig.update_layout(title="<b>ESG Risk — Return Carbon + Deadstock</b>",
                      height=560, showlegend=False)
    export_fig(fig, "43_esg_risks")
    return fig


# =============================================================================
# 21. ACT 12 — DECISION SUPPORT (v4 strategist follow-up)
# =============================================================================
def chart_44_value_at_stake(tx, cust, d, metrics):
    """Quantified Value-at-Stake table — McKinsey style.

    Each recommendation is sized in VND of annual uplift / saved cost,
    paired with a rough investment estimate and payback months.
    """
    recs = []
    nr = metrics["net_revenue"]

    # ------------------------------------------------------------------
    # 1. Cut bottom-quintile promo campaigns
    # ------------------------------------------------------------------
    p_agg = (tx[tx["promo_id"].notna()]
              .groupby("promo_id")
              .agg(revenue=("net_revenue","sum"),
                   discount=("discount_amount","sum")))
    p_agg["roi"] = p_agg["revenue"] / p_agg["discount"].clip(lower=1)
    bottom = p_agg[p_agg["roi"] <= p_agg["roi"].quantile(0.20)]
    # Conservative: 50% of customers buy at full price even without promo
    promo_cut_savings = bottom["discount"].sum() * 0.5
    recs.append(dict(
        action="Cut bottom-quintile promo campaigns (low ROI)",
        annual_uplift=promo_cut_savings,
        investment=10_000_000,    # analyst hours
        owner="Marketing", risk="Low"))

    # ------------------------------------------------------------------
    # 2. Reduce COD cancel rate by 50%
    # ------------------------------------------------------------------
    orders = d["orders"]
    cod = orders[orders["payment_method"] == "cod"]
    n_cancel_cod = (cod["order_status"] == "cancelled").sum()
    # Assume each cancelled COD wastes shipping cost = 10% of avg basket
    avg_basket = nr / max(metrics["n_orders"], 1)
    cod_savings = n_cancel_cod * 0.5 * avg_basket * 0.10
    recs.append(dict(
        action="Reduce COD cancel rate 50% (deposit / OTP verify)",
        annual_uplift=cod_savings,
        investment=200_000_000,
        owner="Ops", risk="Medium"))

    # ------------------------------------------------------------------
    # 3. Wardrobing flag — top 1% serial returners
    # ------------------------------------------------------------------
    cr = (tx.groupby("customer_id")
            .agg(rr=("is_returned","mean"),
                 orders_n=("order_id","nunique"),
                 refunds=("refund_amt","sum")).reset_index())
    wardrobers = cr[(cr["rr"] > BENCHMARKS["wardrobing_return_rate"]) &
                    (cr["orders_n"] >= 3)]
    wardrobing_savings = wardrobers["refunds"].sum() * 0.7
    recs.append(dict(
        action="Wardrobing flag program (>50% RR, ≥3 orders)",
        annual_uplift=wardrobing_savings,
        investment=50_000_000,
        owner="Returns", risk="Low"))

    # ------------------------------------------------------------------
    # 4. Eliminate stockout on top-20 SKUs
    # ------------------------------------------------------------------
    inv = d["inventory"].copy()
    avg_price = tx.groupby("product_id")["unit_price"].mean()
    inv = inv.merge(avg_price.rename("avg_price"), on="product_id", how="left")
    inv["lost"] = inv["stockout_days"] * (inv["units_sold"]/30) * inv["avg_price"]
    top20_lost = inv.groupby("product_id")["lost"].sum().nlargest(20).sum()
    stockout_savings = top20_lost * 0.5  # conservative recovery
    recs.append(dict(
        action="Eliminate stockout on top-20 SKUs",
        annual_uplift=stockout_savings,
        investment=500_000_000,
        owner="Merch", risk="Medium"))

    # ------------------------------------------------------------------
    # 5. Markdown deadstock (>180 days, low STR)
    # ------------------------------------------------------------------
    launch = tx.groupby("product_id")["order_date"].min().rename("launch")
    latest = (d["inventory"].sort_values("snapshot_date")
                             .groupby("product_id").tail(1)
                             .merge(launch.reset_index(),on="product_id",how="left"))
    cogs_map = tx[["product_id","cogs"]].drop_duplicates("product_id")
    latest = latest.merge(cogs_map, on="product_id", how="left", suffixes=("","_c"))
    ccol = "cogs_c" if "cogs_c" in latest.columns else "cogs"
    latest["age"] = (latest["snapshot_date"] - latest["launch"]).dt.days
    deadstock_value = latest.loc[(latest["age"] > BENCHMARKS["deadstock_days"]) &
                                  (latest["sell_through_rate"] < 0.3),
                                  "stock_on_hand"] * latest[ccol]
    deadstock_recovery = deadstock_value.fillna(0).sum() * 0.4   # 40% liquidation
    recs.append(dict(
        action="Liquidate deadstock (age >180d, STR <30%)",
        annual_uplift=deadstock_recovery,
        investment=20_000_000,
        owner="Merch", risk="Low"))

    # ------------------------------------------------------------------
    # 6. Upsell to single-category customers (cross-sell to 2nd category)
    # ------------------------------------------------------------------
    single_cat = cust[cust["categories"] == 1]
    avg_aov = cust["aov"].median()
    # Conservative: 10% of single-cat customers add 1 order @ AOV
    upsell_uplift = len(single_cat) * 0.10 * avg_aov
    recs.append(dict(
        action="Cross-sell single-category customers (Kumar & Shah 2004)",
        annual_uplift=upsell_uplift,
        investment=80_000_000,
        owner="CRM", risk="Low"))

    # ------------------------------------------------------------------
    # 7. Renegotiate supplier on negative-margin SKUs
    # ------------------------------------------------------------------
    prods = d["products"]
    bad_skus = prods[prods["cogs"] / prods["price"].clip(lower=1) > 0.85]
    bad_volume = (tx[tx["product_id"].isin(bad_skus["product_id"])]
                    ["net_revenue"].sum())
    supplier_savings = bad_volume * 0.10  # 10pp margin recovery if renegotiated
    recs.append(dict(
        action="Renegotiate supplier on cogs/price >85% SKUs",
        annual_uplift=supplier_savings,
        investment=30_000_000,
        owner="Procurement", risk="Medium"))

    df = pd.DataFrame(recs)
    df["annual_uplift_M"] = df["annual_uplift"] / 1_000_000
    df["investment_M"]    = df["investment"]    / 1_000_000
    df["payback_months"]  = (df["investment"] /
                              df["annual_uplift"].clip(lower=1) * 12).clip(0.1, 60)
    df = df.sort_values("annual_uplift_M", ascending=True)
    total_uplift_m = df["annual_uplift_M"].sum()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=[
                            f"Annual uplift / saving (M VND)  ·  "
                            f"Total = {total_uplift_m:,.0f}M",
                            "Payback (months) — log scale"])

    risk_color = {"Low":PALETTE["primary"], "Medium":PALETTE["accent"],
                  "High":PALETTE["danger"]}
    colors = [risk_color.get(r, PALETTE["muted"]) for r in df["risk"]]
    fig.add_trace(go.Bar(x=df["annual_uplift_M"], y=df["action"],
                         orientation="h", marker_color=colors,
                         text=df["annual_uplift_M"].round(0).astype(int).astype(str)
                              + "M", textposition="outside",
                         hovertext=df["owner"], showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=df["payback_months"], y=df["action"],
                         orientation="h",
                         marker_color=PALETTE["info"],
                         text=df["payback_months"].round(1),
                         textposition="outside",
                         showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_layout(title="<b>Value-at-Stake — Quantified Recommendations</b>",
                      height=540)
    export_fig(fig, "44_value_at_stake")
    # Persist for MD report
    df.to_parquet(OUTPUT_DIR / "value_at_stake.parquet", index=False)
    return fig, df


def chart_45_clv_churn(cust, tx):
    """Simplified probabilistic CLV (BG/NBD-flavoured, no extra deps).

    P(alive | recency, frequency) ≈ exp(−recency / (k · avg_gap))
    Expected revenue 12m = freq_per_month × 12 × P(alive) × AOV
    Churn score = 1 − P(alive)
    """
    c = cust.copy()
    c["months_active"]   = (c["tenure_days"].clip(lower=1)) / 30 + 1
    c["freq_per_month"]  = c["orders"] / c["months_active"]
    c["avg_gap_days"]    = c["months_active"] * 30 / c["orders"].clip(lower=1)
    decay_window = c["avg_gap_days"] * 2 + 30
    c["p_alive"]         = np.exp(-c["recency_days"] / decay_window).clip(0, 1)
    c["churn_score"]     = 1 - c["p_alive"]
    c["exp_orders_12m"]  = c["freq_per_month"] * 12 * c["p_alive"]
    c["exp_revenue_12m"] = c["exp_orders_12m"] * c["aov"]

    total_exp_rev = c["exp_revenue_12m"].sum()
    median_palive = c["p_alive"].median()
    high_churn_pct = (c["churn_score"] > 0.7).mean()

    fig = make_subplots(rows=2, cols=2,
                        column_widths=[0.55, 0.45],
                        subplot_titles=[
                            "P(alive) distribution",
                            "Churn-score distribution",
                            "Expected 12-month revenue per customer (top 100)",
                            "Aggregate expected revenue by RFM segment"])

    # P(alive) hist
    fig.add_trace(go.Histogram(x=c["p_alive"], nbinsx=40,
                               marker_color=PALETTE["primary"],
                               showlegend=False), row=1, col=1)
    fig.add_vline(x=median_palive, line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"median {median_palive:.2f}",
                  row=1, col=1)
    # Churn hist
    fig.add_trace(go.Histogram(x=c["churn_score"], nbinsx=40,
                               marker_color=PALETTE["danger"],
                               showlegend=False), row=1, col=2)
    fig.add_vline(x=0.7, line_dash="dash", line_color=PALETTE["muted"],
                  annotation_text=f"high-churn {high_churn_pct:.1%}",
                  row=1, col=2)
    # Top 100 expected rev
    top100 = c.nlargest(100, "exp_revenue_12m")
    fig.add_trace(go.Scatter(x=list(range(1,101)), y=top100["exp_revenue_12m"],
                             mode="lines+markers",
                             line=dict(color=PALETTE["primary"], width=1),
                             marker=dict(size=4),
                             showlegend=False),
                  row=2, col=1)
    # By RFM segment
    seg = (c.groupby("rfm_segment")["exp_revenue_12m"].sum()
            .sort_values(ascending=True))
    fig.add_trace(go.Bar(x=seg.values, y=seg.index, orientation="h",
                         marker_color=[PALETTE["seg"].get(s, PALETTE["muted"])
                                       for s in seg.index],
                         text=[f"{v/1e6:,.0f}M" for v in seg.values],
                         textposition="outside",
                         showlegend=False),
                  row=2, col=2)

    fig.update_layout(
        title=f"<b>Probabilistic CLV + Churn Distribution</b>  ·  "
              f"Total expected 12m revenue = {total_exp_rev/1e6:,.0f}M VND  ·  "
              f"high-churn customers = {high_churn_pct:.1%}",
        height=720)
    fig.update_xaxes(title_text="P(alive)",     row=1, col=1)
    fig.update_xaxes(title_text="churn score",  row=1, col=2)
    fig.update_xaxes(title_text="rank",         row=2, col=1)
    fig.update_yaxes(title_text="expected rev", row=2, col=1)
    export_fig(fig, "45_clv_churn")
    return fig, c


def chart_46_activation_funnel(d):
    """Activation funnel — Signup → First Realized Order, broken out by channel.

    Per consistency-audit Finding 3: 3-way split (realized active /
    cancelled-only / silent), not 2-way as v3-v5 RFM showed.
    """
    cu = d["customers"].copy()
    od = d["orders"]
    # First REALIZED (non-cancelled) order — per [A4_active_customer_def]
    od_real = od[od["order_status"] != "cancelled"]
    first_real  = od_real.groupby("customer_id")["order_date"].min().rename("first_real")
    first_any   = od.groupby("customer_id")["order_date"].min().rename("first_any")
    cu = cu.merge(first_real.reset_index(), on="customer_id", how="left")
    cu = cu.merge(first_any.reset_index(),  on="customer_id", how="left")
    cu["realized_active"] = cu["first_real"].notna()
    cu["any_order"]       = cu["first_any"].notna()
    cu["days_to_activate"] = (cu["first_real"] - cu["signup_date"]).dt.days

    def bucket(row):
        if not row["any_order"]:                return "Silent (0 orders)"
        if not row["realized_active"]:          return "Cancelled-only (no realized)"
        d = row["days_to_activate"]
        if pd.isna(d):                          return "Cancelled-only (no realized)"
        if d < 0:                               return "Pre-signup (data integrity)"
        if d <= 7:                              return "Within 7 days"
        if d <= 30:                             return "Within 30 days"
        if d <= 90:                             return "Within 90 days"
        return "Beyond 90 days"

    cu["activation_bucket"] = cu.apply(bucket, axis=1)
    funnel = (pd.crosstab(cu["acquisition_channel"], cu["activation_bucket"],
                          normalize="index") * 100)

    bucket_order = ["Within 7 days","Within 30 days","Within 90 days",
                    "Beyond 90 days","Pre-signup (data integrity)",
                    "Cancelled-only (no realized)","Silent (0 orders)"]
    bucket_colors = ["#1F7A4C","#52BE80","#F5CBA7","#E8A33D","#85929E",
                     "#E67E22","#C0392B"]

    activated_buckets = [b for b in bucket_order
                         if b not in ("Silent (0 orders)","Cancelled-only (no realized)")
                         and b in funnel.columns]
    funnel["_activation"] = funnel[activated_buckets].sum(axis=1)
    funnel = funnel.sort_values("_activation", ascending=True)
    realized_pct = cu["realized_active"].mean() * 100
    n_realized   = int(cu["realized_active"].sum())
    n_cancelled_only = int(cu["any_order"].sum() - cu["realized_active"].sum())
    n_silent     = int((~cu["any_order"]).sum())

    fig = go.Figure()
    for b, c in zip(bucket_order, bucket_colors):
        if b in funnel.columns:
            fig.add_trace(go.Bar(name=b, y=funnel.index, x=funnel[b],
                                 orientation="h", marker_color=c))
    fig.add_vline(x=realized_pct, line_dash="dash", line_color="black",
                  annotation_text=f"Overall realized {realized_pct:.1f}%",
                  annotation_position="top right")
    fig.update_layout(
        barmode="stack",
        title=f"<b>Activation Funnel</b> — 3-way: realized {n_realized:,} "
              f"({realized_pct:.2f}%) · cancelled-only {n_cancelled_only:,} "
              f"({n_cancelled_only/len(cu)*100:.2f}%) · silent {n_silent:,} "
              f"({n_silent/len(cu)*100:.2f}%)",
        xaxis_title="% of Signups",
        yaxis_title="Acquisition Channel",
        legend_title="Outcome",
        height=560, xaxis_range=[0, 100])
    export_fig(fig, "46_activation_funnel")
    return fig


# =============================================================================
# 21b. ACT 12 — Diagnostic plots A/B/C (v6 strategist follow-up)
# =============================================================================
def chart_47_signup_gap_distribution(d):
    """Plot A — Signup gap distribution diagnostic for [signup_after_first_order].

    Histogram of (first_order_date - signup_date) in days. Helps localize root
    cause of the 89.34% violation rate:
      - gap median 1-7d  → guest-checkout retroactive registration
      - gap median 30-90d → account migration
      - gap median in years → simulator artifact
    """
    cu = d["customers"].copy()
    od = d["orders"]
    first_order = od.groupby("customer_id")["order_date"].min().rename("first_order")
    cu = cu.merge(first_order.reset_index(), on="customer_id", how="inner")
    cu["gap_days"] = (cu["first_order"] - cu["signup_date"]).dt.days

    n_violation = int((cu["gap_days"] < 0).sum())
    n_total = len(cu)
    pct_violation = n_violation / n_total * 100
    median_gap_violators = float(cu.loc[cu["gap_days"] < 0, "gap_days"].median())

    fig = go.Figure()
    cu_norm = cu[cu["gap_days"] >= 0]
    cu_anom = cu[cu["gap_days"] < 0]
    fig.add_trace(go.Histogram(x=cu_norm["gap_days"], nbinsx=80,
                                marker_color=PALETTE["primary"],
                                name=f"Normal (gap ≥ 0): {len(cu_norm):,}"))
    fig.add_trace(go.Histogram(x=cu_anom["gap_days"], nbinsx=80,
                                marker_color=PALETTE["danger"],
                                name=f"Anomaly (gap < 0): {len(cu_anom):,}"))
    for x, lbl in [(1,"1d"),(7,"7d"),(30,"30d"),(365,"1y"),(-365,"-1y"),(-30,"-30d")]:
        fig.add_vline(x=x, line_dash="dot", line_color=PALETTE["muted"],
                      annotation_text=lbl, annotation_position="top")
    fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"<b>Signup Gap Distribution</b> — diagnostic for "
              f"`signup_after_first_order`  ·  {pct_violation:.2f}% violation "
              f"(median gap of violators = {median_gap_violators:.0f} days)",
        xaxis_title="gap days = first_order_date − signup_date",
        yaxis_title="customers (log scale)",
        barmode="overlay", height=520)
    export_fig(fig, "47_signup_gap_distribution")
    return fig


def chart_48_cancelled_only_profile(d):
    """Plot B — Cancelled-only segment profile (4-panel).

    Diagnostic for the 2,123 `cancelled_only_segment`. Compares segment vs
    overall population on payment_method, acquisition_channel, region, cancel
    reason → identifies fraud/COD-trap clusters.
    """
    od = d["orders"]; cu = d["customers"]; geo = d["geography"]
    # Identify cancelled-only customers
    statuses_per_cust = od.groupby("customer_id")["order_status"].apply(set)
    co_ids = statuses_per_cust[statuses_per_cust.apply(
        lambda s: ("cancelled" in s) and not (s & {"delivered","returned"}))].index
    co_orders = od[od["customer_id"].isin(co_ids)]
    co_cust   = cu[cu["customer_id"].isin(co_ids)]

    n_co = len(co_ids)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Payment method ({n_co:,} cancelled-only vs all)",
            f"Acquisition channel ({n_co:,} vs all)",
            "Region (vs all)",
            "Cancel reason proxy (status ≠ delivered/returned)"
        ])

    def _norm_compare(co_series, all_series, panel_row, panel_col):
        co_pct  = (co_series.value_counts(normalize=True)*100).rename("co")
        all_pct = (all_series.value_counts(normalize=True)*100).rename("all")
        merged = pd.concat([co_pct, all_pct], axis=1).fillna(0).sort_values("co", ascending=True)
        fig.add_trace(go.Bar(y=merged.index, x=merged["co"], name="Cancelled-only",
                             orientation="h", marker_color=PALETTE["danger"],
                             showlegend=(panel_row==1 and panel_col==1)),
                      row=panel_row, col=panel_col)
        fig.add_trace(go.Bar(y=merged.index, x=merged["all"], name="All customers",
                             orientation="h", marker_color=PALETTE["primary"],
                             showlegend=(panel_row==1 and panel_col==1)),
                      row=panel_row, col=panel_col)

    _norm_compare(co_orders["payment_method"], od["payment_method"], 1, 1)
    _norm_compare(co_cust["acquisition_channel"], cu["acquisition_channel"], 1, 2)

    # Region — join orders.zip → geography
    co_orders_g = co_orders.merge(geo[["zip","region"]], on="zip", how="left")
    od_g        = od.merge(geo[["zip","region"]], on="zip", how="left")
    _norm_compare(co_orders_g["region"], od_g["region"], 2, 1)

    # Status mix among cancelled-only customers' orders
    _norm_compare(co_orders["order_status"], od["order_status"], 2, 2)

    # Highlight any > 5pp deviation from baseline
    fig.update_layout(
        title=f"<b>Cancelled-only Segment Profile</b> — {n_co:,} customers "
              f"({n_co/len(cu)*100:.2f}%) · red bars > blue = over-represented = "
              "fraud / COD-trap signal",
        barmode="group", height=720,
        legend=dict(orientation="h", y=1.08))
    export_fig(fig, "48_cancelled_only_profile")
    return fig


def chart_49_activation_clv_overlay(d):
    """Plot C — Activation rate × Median CLV by acquisition channel.

    Extends Chart 46. Bar = activation rate (within 30 days); line = median
    revenue per realized active customer. Reveals CAC efficiency tradeoffs:
    high-volume low-conversion channels ≠ high-volume high-conversion ones.
    """
    od = d["orders"]; cu = d["customers"]
    # Realized active customer revenue
    od_real = od[od["order_status"].isin(["delivered","returned"])]
    realized_by_cust = od_real.groupby("customer_id").size().index
    cu_active = cu[cu["customer_id"].isin(realized_by_cust)].copy()

    first_real = od_real.groupby("customer_id")["order_date"].min().rename("first_real")
    cu_active = cu_active.merge(first_real.reset_index(), on="customer_id", how="left")
    cu_active["days_to_activate"] = (cu_active["first_real"] - cu_active["signup_date"]).dt.days

    # Per-channel: # customers who activated within 30 days of signup
    cu["realized_active"] = cu["customer_id"].isin(realized_by_cust)
    cu_active_30d = cu_active[(cu_active["days_to_activate"] >= 0) &
                                (cu_active["days_to_activate"] <= 30)]
    activation_30d = (cu_active_30d.groupby("acquisition_channel").size() /
                      cu.groupby("acquisition_channel").size() * 100)

    # CLV per channel (median revenue of realized active customers)
    if "tx_master.parquet" in {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}:
        tx = pd.read_parquet(OUTPUT_DIR / "tx_master.parquet")
        tx_real = tx[tx["is_cancelled"]==0]
        clv = tx_real.groupby("customer_id")["net_revenue"].sum()
        cu_active = cu_active.merge(clv.rename("clv").reset_index(),
                                     on="customer_id", how="left")
        median_clv = cu_active.groupby("acquisition_channel")["clv"].median()
    else:
        # Fallback: payment_value
        pay = d["payments"].merge(od[["order_id","customer_id","order_status"]],
                                   on="order_id")
        pay_real = pay[pay["order_status"]!="cancelled"]
        clv = pay_real.groupby("customer_id")["payment_value"].sum()
        cu_active = cu_active.merge(clv.rename("clv").reset_index(),
                                     on="customer_id", how="left")
        median_clv = cu_active.groupby("acquisition_channel")["clv"].median()

    df = pd.concat([activation_30d.rename("act_30d_pct"),
                    median_clv.rename("median_clv")], axis=1).dropna()
    df = df.sort_values("act_30d_pct", ascending=True)

    # Color: <50% = high-risk (red), 50-70% = warning (amber), ≥70% = healthy (green)
    bar_colors = [PALETTE["danger"] if v < 50 else
                  PALETTE["accent"] if v < 70 else PALETTE["primary"]
                  for v in df["act_30d_pct"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["act_30d_pct"],
                         marker_color=bar_colors,
                         text=[f"{v:.1f}%" for v in df["act_30d_pct"]],
                         textposition="outside",
                         name="30-day activation rate %"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["median_clv"],
                             mode="lines+markers",
                             line=dict(color=PALETTE["info"], width=3),
                             marker=dict(size=10),
                             name="Median CLV (realized active)"),
                  secondary_y=True)
    fig.add_hline(y=50, line_dash="dash", line_color=PALETTE["danger"],
                  secondary_y=False,
                  annotation_text="<50% = high-risk acquisition")
    fig.update_layout(
        title=f"<b>Activation Rate × CLV by Channel</b> — CAC efficiency map  ·  "
              "bar = 30-day activation; line = median CLV of realized actives",
        height=520, barmode="group",
        legend=dict(orientation="h", y=1.10))
    fig.update_xaxes(title_text="Acquisition Channel")
    fig.update_yaxes(title_text="30-day activation %", secondary_y=False)
    fig.update_yaxes(title_text="Median CLV (VND scaled)", secondary_y=True)
    export_fig(fig, "49_activation_clv_overlay")
    return fig


# =============================================================================
# 21c. ACT 12 — Phantom catalog + Scenario projection (v6 follow-up)
# =============================================================================
def chart_50_phantom_catalog_composition(d, tx):
    """Phantom catalog composition — 814 unsold vs 1,598 sold SKUs.

    3 panels:
      - Donut: SKU count split (sold / phantom unsold)
      - Phantom % by category (which categories are most padded)
      - List margin distribution comparison (sold vs phantom)
    """
    sold_skus = set(tx["product_id"].unique())
    products = d["products"].copy()
    products["status"] = products["product_id"].apply(
        lambda p: "Sold" if p in sold_skus else "Phantom (unsold)")
    products["list_margin"] = ((products["price"] - products["cogs"])
                                / products["price"] * 100)

    counts = products["status"].value_counts()
    n_phantom = int((products["status"] == "Phantom (unsold)").sum())
    n_total = len(products)

    by_cat = pd.crosstab(products["category"], products["status"])
    if "Phantom (unsold)" not in by_cat.columns:
        by_cat["Phantom (unsold)"] = 0
    by_cat["phantom_pct"] = by_cat["Phantom (unsold)"] / by_cat.sum(axis=1) * 100
    by_cat = by_cat.sort_values("phantom_pct", ascending=True)

    margin_sold    = products[products["status"] == "Sold"]["list_margin"]
    margin_phantom = products[products["status"] == "Phantom (unsold)"]["list_margin"]

    fig = make_subplots(rows=1, cols=3,
                        column_widths=[0.28, 0.36, 0.36],
                        specs=[[{"type":"domain"},
                                {"type":"xy"},
                                {"type":"xy"}]],
                        subplot_titles=[
                            f"SKU split (n={n_total:,})",
                            "Phantom % per category",
                            "List margin distribution (sold vs phantom)"])

    # Panel 1: Donut
    fig.add_trace(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                         marker=dict(colors=[PALETTE["primary"], PALETTE["danger"]]),
                         textposition="inside",
                         textinfo="label+percent+value",
                         showlegend=False),
                  row=1, col=1)

    # Panel 2: Phantom % per category
    fig.add_trace(go.Bar(x=by_cat["phantom_pct"], y=by_cat.index,
                         orientation="h",
                         marker_color=PALETTE["danger"],
                         text=[f"{v:.1f}%" for v in by_cat["phantom_pct"]],
                         textposition="outside",
                         showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(title_text="% phantom (unsold)", range=[0, 100], row=1, col=2)

    # Panel 3: Margin histograms (overlapping)
    fig.add_trace(go.Histogram(x=margin_sold, nbinsx=30,
                                name=f"Sold (median {margin_sold.median():.1f}%)",
                                marker_color=PALETTE["primary"], opacity=0.65),
                  row=1, col=3)
    fig.add_trace(go.Histogram(x=margin_phantom, nbinsx=30,
                                name=f"Phantom (median {margin_phantom.median():.1f}%)",
                                marker_color=PALETTE["danger"], opacity=0.65),
                  row=1, col=3)
    fig.update_xaxes(title_text="List margin %", row=1, col=3)
    fig.update_yaxes(title_text="SKU count", row=1, col=3)

    fig.update_layout(
        title=(f"<b>Phantom Catalog Composition</b> — "
               f"{n_phantom}/{n_total:,} SKUs ({n_phantom/n_total*100:.1f}%) "
               f"never sold  ·  phantom median list margin "
               f"{margin_phantom.median():.1f}% vs sold "
               f"{margin_sold.median():.1f}% "
               f"(+{margin_phantom.median()-margin_sold.median():.1f}pp gap → "
               "selection bias toward high-margin SKUs sitting unsold)"),
        barmode="overlay", height=560,
        legend=dict(orientation="h", y=-0.18))
    export_fig(fig, "50_phantom_catalog_composition")
    return fig


def chart_51_scenario_projection(tx, d, metrics):
    """v6 scenario projection — CM trajectory under 4 operational levers.

    Each scenario applies one operational change with documented assumption.
    Scenarios A/B/C are independent; D combines all (additive — no interaction
    effects modeled). Conservative assumptions throughout; see assumption
    column for full math.
    """
    nr_base       = metrics["net_revenue"]
    cogs_base     = metrics["total_cogs"]
    var_base      = (metrics["payment_fee"] + metrics["packaging_cost"]
                     + metrics["return_proc_cost"] + metrics["total_refund"])
    cm_base       = metrics["contribution_margin"]
    cm_pct_base   = metrics["contribution_margin_pct"] * 100
    cogs_rate     = cogs_base / nr_base
    var_rate      = var_base  / nr_base
    cm_rate_base  = cm_base   / nr_base

    rows = []

    def _record(name, nr_lift, cm_lift, assumption):
        new_nr = nr_base + nr_lift
        new_cm_abs = cm_base + cm_lift
        new_cm_pct = new_cm_abs / new_nr * 100
        rows.append(dict(
            scenario=name,
            nr_lift=nr_lift, cm_lift=cm_lift,
            new_cm_abs=new_cm_abs, new_cm_pct=new_cm_pct,
            delta_pp=new_cm_pct - cm_pct_base,
            assumption=assumption,
        ))

    # Baseline
    _record("Baseline (current)", 0, 0, "As-is per validation_v6 (verified)")

    # ------ Scenario A — Activate 50% of phantom catalog ------
    sold_skus = set(tx["product_id"].unique())
    products = d["products"]
    phantom = products[~products["product_id"].isin(sold_skus)]
    sold_p   = products[products["product_id"].isin(sold_skus)]
    n_activated = int(len(phantom) * 0.50)
    rev_per_sold_sku = nr_base / max(len(sold_p), 1)
    velocity = 0.30                                   # 30% of sold-cohort velocity
    nr_lift_A = n_activated * rev_per_sold_sku * velocity
    phantom_cogs_ratio = (phantom["cogs"] / phantom["price"]).mean()
    cogs_lift_A = nr_lift_A * phantom_cogs_ratio
    var_lift_A = nr_lift_A * var_rate
    cm_lift_A = nr_lift_A - cogs_lift_A - var_lift_A
    _record(f"A — Activate 50% phantom ({n_activated} SKUs)",
            nr_lift_A, cm_lift_A,
            f"@ {velocity:.0%} of sold velocity; phantom cogs/price={phantom_cogs_ratio:.2f}")

    # ------ Scenario B — Halve COD cancel rate ------
    od = d["orders"]
    n_cod_cancel = int(((od["payment_method"] == "cod") &
                         (od["order_status"] == "cancelled")).sum())
    n_saved = n_cod_cancel // 2
    avg_basket = nr_base / max(metrics["n_orders"], 1)
    nr_lift_B = n_saved * avg_basket
    cogs_lift_B = nr_lift_B * cogs_rate
    var_lift_B = nr_lift_B * var_rate
    cm_lift_B = nr_lift_B - cogs_lift_B - var_lift_B
    _record(f"B — Halve COD cancel ({n_saved:,} orders saved)",
            nr_lift_B, cm_lift_B,
            "Saved cancels convert at avg basket × baseline margin profile")

    # ------ Scenario C — Cut bottom-quintile promos ------
    pu = (tx[tx["promo_id"].notna()].groupby("promo_id")
                                     .agg(revenue=("net_revenue","sum"),
                                          discount=("discount_amount","sum")))
    pu["roi"] = pu["revenue"] / pu["discount"].clip(lower=1)
    bottom_q = pu[pu["roi"] <= pu["roi"].quantile(0.20)]
    cannibal_rate = 0.30
    revenue_lost = bottom_q["revenue"].sum() * cannibal_rate
    discount_saved = bottom_q["discount"].sum()
    nr_lift_C = -revenue_lost
    cm_lift_C = -revenue_lost * cm_rate_base + discount_saved
    n_promos_cut = len(bottom_q)
    _record(f"C — Kill bottom-quintile promos ({n_promos_cut} campaigns)",
            nr_lift_C, cm_lift_C,
            f"30% cannibalization; saves {discount_saved/1e6:,.0f}M VND discount")

    # ------ Scenario D — All combined (additive) ------
    nr_lift_D = nr_lift_A + nr_lift_B + nr_lift_C
    cm_lift_D = cm_lift_A + cm_lift_B + cm_lift_C
    _record("D — All combined (A + B + C)",
            nr_lift_D, cm_lift_D,
            "Additive — no interaction effects modeled (conservative)")

    df = pd.DataFrame(rows)

    # 2-panel: bar of CM trajectory + table of details
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        specs=[[{"type":"bar"}, {"type":"table"}]],
                        subplot_titles=[
                            "CM trajectory by scenario",
                            "Projection details"])

    bar_colors = []
    for r in rows:
        if r["delta_pp"] == 0:
            bar_colors.append(PALETTE["muted"])
        elif r["delta_pp"] > 0:
            bar_colors.append(PALETTE["primary"])
        else:
            bar_colors.append(PALETTE["danger"])

    short_names = [r["scenario"][:35] for r in rows]
    fig.add_trace(go.Bar(x=short_names, y=df["new_cm_pct"],
                         marker_color=bar_colors,
                         text=[f"{v:+.2f}%" for v in df["new_cm_pct"]],
                         textposition="outside",
                         showlegend=False), row=1, col=1)
    fig.add_hline(y=cm_pct_base, line_dash="dash", line_color=PALETTE["muted"],
                  annotation_text=f"Baseline {cm_pct_base:+.2f}%",
                  row=1, col=1)
    fig.update_yaxes(title_text="Goods-only CM %", row=1, col=1)

    fig.add_trace(go.Table(
        columnwidth=[40, 18, 18, 14, 80],
        header=dict(values=["Scenario", "ΔNR (M)", "CM new", "ΔPP",
                            "Assumption"],
                    fill_color=PALETTE["primary"],
                    font=dict(color="white", size=11),
                    align="left"),
        cells=dict(values=[
            df["scenario"],
            [f"{v/1e6:+,.0f}" for v in df["nr_lift"]],
            [f"{v:+.2f}%" for v in df["new_cm_pct"]],
            [f"{v:+.2f}" for v in df["delta_pp"]],
            df["assumption"]
        ], align="left", font=dict(size=10))),
        row=1, col=2)

    final_row = rows[-1]
    fig.update_layout(
        title=(f"<b>Scenario Projection — Goods-only CM trajectory</b>  ·  "
               f"baseline {cm_pct_base:+.2f}% → combined "
               f"{final_row['new_cm_pct']:+.2f}% "
               f"(Δ {final_row['delta_pp']:+.2f}pp)"),
        height=560)
    export_fig(fig, "51_scenario_projection")
    # Persist for downstream reporting
    df.to_parquet(OUTPUT_DIR / "scenario_projection.parquet", index=False)
    return fig, df


# =============================================================================
# 22. MARKDOWN REPORT BUILDER
# =============================================================================
def _fmt_vnd(x): return f"{x:,.0f} VND"
def _fmt_pct(x): return f"{x*100:.2f}%"

def build_markdown_report(metrics, out_path: Path):
    m = metrics
    a = []

    # ---------- Header ----------
    a.append("# DATATHON 2026 — THE GRIDBREAKER")
    a.append("## Enterprise-grade EDA Report · Fashion E-commerce")
    a.append("*VinTelligence — VinUniversity DS&AI Club*")
    a.append("")
    a.append(f"**Window**: `{m['period_start'].date()}` → `{m['period_end'].date()}`  ")
    a.append(f"**Charts**: 52 PNGs in `./figures/` · **Methodology**: "
             "industry-benchmarked, research-referenced")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 0. HEADLINE FINDINGS (v6 — reframed per A9_shipping_treatment) ----------
    cm_pct = m["contribution_margin_pct"] * 100
    a.append("## 🚨 HEADLINE FINDINGS — Read everything else through these")
    a.append("")
    a.append("### Finding 1 — Goods-only contribution margin is structurally thin")
    a.append("")
    a.append(f"> **Contribution Margin (goods-only)** = **{cm_pct:+.2f}%** of net revenue "
             f"({_fmt_vnd(m['contribution_margin'])}).")
    a.append(">")
    a.append("> **Goods-only basis explained** (per `[A9_shipping_treatment]`):")
    a.append(">")
    a.append("> CM reported on goods-only basis. Dataset's `shipping_fee` is "
             "customer-paid fee (revenue-side), not carrier cost (which is not in "
             "dataset). Including shipping with hypothetical carrier cost would "
             "produce a wide CM range (−10% to +2%) speculative on cost not "
             "observed. **Excluding shipping prioritizes data integrity**; matches "
             "Inditex/Shein practice of separating logistics from gross margin "
             "reporting. True business CM including logistics P&L cannot be "
             "computed from this dataset.")
    a.append(">")
    a.append("> **Strategic reading**: At +2% goods-only, every variable cost line "
             "(payment fees, packaging, returns) compresses margin to near-zero "
             "before logistics. This is a *growth-trap signal* (Christensen 1997 "
             "*Innovator's Dilemma*) — if true logistics cost is anything close to "
             "industry benchmark (8-12%), high retention amplifies losses rather "
             "than profits. Pattern matches WeWork / Casper / Blue Apron.")
    a.append("")
    a.append("### Finding 2 — Headline KPIs were over-stated by ~9% in v3")
    a.append("")
    a.append(f"> Per `[order_status_in_oi]` assumption: **{m['n_cancelled']:,}** "
             f"({m['cancel_rate']*100:.2f}% of all orders) have `order_status='cancelled'` "
             "**but still appear in `order_items`**. v5 (this report) filters them out "
             f"per `[A3_realized_order_def]`. Net revenue = "
             f"{_fmt_vnd(m['net_revenue'])}; without the filter v3 reported a higher number.")
    a.append("")
    silent_pct = m["n_silent_strict"] / m["n_customers_total"] * 100
    cancelled_only_pct = m["n_cancelled_only"] / m["n_customers_total"] * 100
    a.append("### Finding 3 — Customer base has 3-way split (not 2-way as v3 RFM showed)")
    a.append("")
    a.append(f"> Per `[silent_cohort]` audit, the {m['n_customers_total']:,} signups "
             "split into 3 groups, not 2:")
    a.append(">")
    a.append(f"> | Cohort | Count | % | v3-v5 RFM treatment |")
    a.append(f"> |---|---|---|---|")
    a.append(f"> | **Realized active** (≥1 non-cancelled order) | "
             f"{m['n_realized_active']:,} | {m['activation_rate']*100:.2f}% | included ✓ |")
    a.append(f"> | **Cancelled-only** (≥1 order, all cancelled) | "
             f"{m['n_cancelled_only']:,} | {cancelled_only_pct:.2f}% | "
             "**invisible** — neither active nor silent |")
    a.append(f"> | **Silent strict** (0 orders ever) | "
             f"{m['n_silent_strict']:,} | {silent_pct:.2f}% | dropped silently |")
    a.append(">")
    a.append(f"> True realized activation rate = **{m['activation_rate']*100:.2f}%** "
             f"(not 100%). The {m['n_cancelled_only']:,} cancelled-only customers are a "
             "separate signal worth tracking — they expressed buying intent then aborted, "
             "different from never-engaged silent customers.")
    a.append("")
    sig_pct = m['signup_after_first_order_pct'] * 100
    a.append(f"### Finding 4 — `signup_date` is **unreliable for {sig_pct:.1f}% of customers** "
             "(data integrity issue)")
    a.append("")
    a.append(f"> Per `[signup_after_first_order]` audit: **{m['n_signup_after_first_order']:,} "
             f"customers ({sig_pct:.2f}%)** "
             "have `signup_date` LATER than their `first_order_date` — physically "
             "impossible. Cohort retention v3 (anchored on signup) is invalid.")
    a.append(">")
    a.append("> v6 fix: cohort heatmap now anchors on **`activation_cohort` = first-order "
             "month**. Tenure metrics also rebuilt from `first_order_date`. The signup_date "
             "field is kept only for descriptive count of total signups (incl. silent).")
    a.append("")
    a.append("### Finding 5 — Forecast target ≠ business-narrative scope (must distinguish)")
    a.append("")
    a.append("> Per `[forecast_target_definition]` audit: **`sales.Revenue` matches "
             "ALL order_items 100%, including cancelled** (H1=100% match, H2=0.2%, H3=0%). "
             "This means:")
    a.append(">")
    a.append("> - **Part 3 forecast model** must train on ALL line values (incl. cancelled). "
             "Filtering cancelled would mismatch the target by 9%.")
    a.append("> - **Part 2 EDA narrative** (margin, customer behavior) uses REALIZED "
             "(excl. cancelled) per `[A3_realized_order_def]` — these answer different "
             "business questions.")
    a.append("> - This report uses REALIZED scope for all KPIs below, with a separate "
             "section noting the 9% gap to forecast scope.")
    a.append("")
    a.append("### How to read every chart below")
    a.append("")
    a.append("- All revenue/margin numbers are **realized only** (cancelled excluded).")
    a.append("- Cohort retention 74% → still impressive but applies only to *activated* "
             "customers, not all signups.")
    a.append("- Promo ROI uses revenue, not CM. To get CM-ROI, scale by gross_margin × CM_factor.")
    a.append("- BCG matrix uses growth × share within already-thin-margin business.")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 1. Executive Summary ----------
    a.append("## 1. Executive Summary")
    a.append("")
    a.append("### Lifetime KPIs")
    a.append("| Metric | Value | Industry Benchmark | Gap |")
    a.append("|---|---|---|---|")
    a.append(f"| Net revenue | {_fmt_vnd(m['net_revenue'])} | — | — |")
    a.append(f"| Gross margin | {_fmt_pct(m['gross_margin'])} | "
             f"{BENCHMARKS['gross_margin_fashion']:.0%} | "
             f"{m['vs_margin_bm']*100:+.1f} pp |")
    a.append(f"| Return rate | {_fmt_pct(m['return_rate'])} | "
             f"{BENCHMARKS['return_rate_online_fashion']:.0%} | "
             f"{m['vs_return_bm']*100:+.1f} pp |")
    a.append(f"| Repeat rate | {_fmt_pct(m['repeat_rate'])} | "
             f"{BENCHMARKS['repeat_rate_12m']:.0%} | "
             f"{m['vs_repeat_bm']*100:+.1f} pp |")
    a.append(f"| Full-price revenue share | {_fmt_pct(m['full_price_rev_share'])} | "
             f"{BENCHMARKS['full_price_sellthrough']:.0%} | "
             f"{(m['full_price_rev_share']-BENCHMARKS['full_price_sellthrough'])*100:+.1f} pp |")
    a.append(f"| SKU Gini | {m['sku_gini']:.3f} | 0.60-0.75 | — |")
    a.append(f"| Customer Gini | {m['cust_gini']:.3f} | 0.50-0.70 | — |")
    a.append(f"| Top-10% customer rev share | {_fmt_pct(m['top10pct_cust_rev_share'])} | "
             f"40-60% | — |")
    a.append("")
    a.append("### ⚠️ Unit economics — goods-only CM per [A9_shipping_treatment]")
    a.append("Gross Profit chưa trừ chi phí biến đổi (payment fees, packaging, "
             "return processing, refunds). **Goods-only Contribution Margin** = "
             "lãi thật sau cửa hàng, EXCLUDING logistics (per A9). "
             "Shipping line shown for reference only — NOT in CM total.")
    a.append("")
    a.append("| Variable cost layer | Value | % of Net Revenue |")
    a.append("|---|---|---|")
    nr = m["net_revenue"]
    a.append(f"| Net Revenue | {_fmt_vnd(nr)} | 100.00% |")
    a.append(f"| − COGS | −{_fmt_vnd(m['total_cogs'])} | {-m['total_cogs']/nr*100:.2f}% |")
    a.append(f"| = Gross Profit | {_fmt_vnd(m['gross_profit'])} | "
             f"{m['gross_margin']*100:.2f}% |")
    a.append(f"| − Payment processing | −{_fmt_vnd(m['payment_fee'])} | "
             f"{-m['payment_fee']/nr*100:.2f}% |")
    a.append(f"| − Packaging | −{_fmt_vnd(m['packaging_cost'])} | "
             f"{-m['packaging_cost']/nr*100:.2f}% |")
    a.append(f"| − Return processing overhead | −{_fmt_vnd(m['return_proc_cost'])} | "
             f"{-m['return_proc_cost']/nr*100:.2f}% |")
    a.append(f"| − Refunds paid out | −{_fmt_vnd(m['total_refund'])} | "
             f"{-m['total_refund']/nr*100:.2f}% |")
    a.append(f"| **= Goods-only Contribution Margin** | "
             f"**{_fmt_vnd(m['contribution_margin'])}** | "
             f"**{m['contribution_margin_pct']*100:+.2f}%** |")
    a.append(f"| *(memo) Shipping fee from customers* | "
             f"*{_fmt_vnd(m['shipping_cost'])}* | "
             f"*{m['shipping_fee_pct_revenue']*100:.3f}%* |")
    a.append("")
    a.append("### Vietnam context snapshot")
    a.append("| Metric | Value | Benchmark |")
    a.append("|---|---|---|")
    a.append(f"| COD share of orders | {_fmt_pct(m['cod_share'])} | "
             f"{BENCHMARKS['cod_share_vn']:.0%} (VN avg) |")
    a.append(f"| Overall cancel rate | {_fmt_pct(m['cancel_rate_overall'])} | "
             f"Non-COD <5% |")
    a.append(f"| COD cancel rate | {_fmt_pct(m['cancel_rate_cod'])} | "
             f"{BENCHMARKS['cod_cancel_rate_vn']:.0%} (Ninja Van 2023) |")
    a.append("")
    a.append("### Pricing paradox check")
    a.append("| Metric | Value |")
    a.append("|---|---|")
    a.append(f"| SKUs with negative list margin (cogs ≥ price) | "
             f"{_fmt_pct(m['sku_negative_margin_pct'])} |")
    a.append(f"| Median list-price margin | "
             f"{_fmt_pct(m['median_list_margin_pct'])} |")
    a.append("")
    a.append("> **Dataset disclaimer (v3)** — AOV ≈ "
             f"{m['aov']:,.0f} VND rất thấp so với fashion VN thực tế "
             "(thường 200-500k VND). Dataset có thể là (a) phụ kiện / "
             "accessory-heavy, (b) đơn vị VND được scale từ gốc USD, hoặc "
             "(c) per-unit thay vì per-order. Mọi KPI tuyệt đối cần được đọc "
             "với caveat này; các ratio / trend / so sánh tương đối vẫn valid.")
    a.append("")
    a.append("### Detailed counts")
    a.append("| Metric | Value |")
    a.append("|---|---|")
    a.append(f"| Total signups | {m['n_customers_total']:,} |")
    a.append(f"| → Realized active (≥1 non-cancelled order) | {m['n_realized_active']:,} "
             f"({m['activation_rate_realized']*100:.2f}%) |")
    a.append(f"| → Cancelled-only (≥1 order, all cancelled) | {m['n_cancelled_only']:,} "
             f"({m['n_cancelled_only']/m['n_customers_total']*100:.2f}%) |")
    a.append(f"| → Silent strict (0 orders ever) | {m['n_silent_strict']:,} "
             f"({m['n_silent_strict']/m['n_customers_total']*100:.2f}%) |")
    a.append(f"| Orders (realized, ex-cancelled) | {m['n_orders']:,} |")
    a.append(f"| Cancelled orders | {m['n_cancelled']:,} "
             f"({m['cancel_rate']*100:.2f}%) |")
    a.append(f"| Products in catalogue | {m['n_products']:,} |")
    a.append(f"| Promotion campaigns | {m['n_promos']:,} |")
    a.append(f"| Gross revenue | {_fmt_vnd(m['gross_revenue'])} |")
    a.append(f"| Total promo discount | {_fmt_vnd(m['total_discount'])} |")
    a.append(f"| Total COGS | {_fmt_vnd(m['total_cogs'])} |")
    a.append(f"| Gross profit | {_fmt_vnd(m['gross_profit'])} |")
    a.append(f"| Total refunds | {_fmt_vnd(m['total_refund'])} |")
    a.append(f"| AOV | {_fmt_vnd(m['aov'])} |")
    a.append(f"| Avg orders / customer | {m['avg_orders_per_cust']:.2f} |")
    a.append(f"| Median recency (days) | {m['median_recency']:.0f} |")
    a.append(f"| Promo-coverage of line-items | {_fmt_pct(m['promo_share'])} |")
    a.append(f"| Full-price line-item share | {_fmt_pct(m['full_price_share'])} |")
    a.append("")
    a.append("### RFM segment distribution")
    a.append("| Segment | Customers |")
    a.append("|---|---|")
    for seg, cnt in sorted(m["segment_counts"].items(), key=lambda kv: -kv[1]):
        a.append(f"| {seg} | {cnt:,} |")
    a.append("")
    a.append("---")

    # ---------- 2. Data Quality Audit ----------
    a.append("## 2. Data Quality Audit — Actual Results (v4)")
    a.append("")
    a.append("Pipeline thực hiện 4 check; bảng dưới là kết quả tính trực tiếp "
             "từ dataset, không phải mô tả lý thuyết.")
    a.append("")
    a.append("### 2.1 Top-8 columns with non-zero null rate")
    a.append("| Table | Column | Null % | Status |")
    a.append("|---|---|---|---|")
    if m["dq_top_nulls"]:
        for r in m["dq_top_nulls"]:
            status = "🔴 HIGH" if r["null_pct"] > 20 else \
                     "🟡 monitor" if r["null_pct"] > 5 else "🟢 OK"
            a.append(f"| `{r['table']}` | `{r['col']}` | "
                     f"{r['null_pct']:.2f}% | {status} |")
    else:
        a.append("| — | — | 0% | 🟢 No nulls detected |")
    a.append("")
    a.append("### 2.2 Referential integrity (orphan FK rows)")
    a.append("| Check | Orphan rows | Status |")
    a.append("|---|---|---|")
    for r in m["dq_integrity"]:
        status = "🟢 PASS" if r["orphan_count"] == 0 else "🔴 FAIL"
        a.append(f"| {r['check']} | {r['orphan_count']:,} | {status} |")
    a.append("")
    a.append("### 2.3 Date coverage")
    a.append("| Table | Start | End |")
    a.append("|---|---|---|")
    for r in m["dq_date_cov"]:
        s = r["start"].date() if hasattr(r["start"], "date") else r["start"]
        e = r["end"].date()   if hasattr(r["end"],   "date") else r["end"]
        a.append(f"| `{r['table']}` | {s} | {e} |")
    a.append("")
    a.append("### 2.4 ⚠ Shipping-fee diagnostic — ROOT of unit-economics paradox")
    sf = m["dq_shipping"]
    a.append("| Stat | Value | Implication |")
    a.append("|---|---|---|")
    a.append(f"| Shipments rows | {sf['n_shipments']:,} | — |")
    a.append(f"| Shipping fee = 0 | **{sf['zero_pct']:.2f}%** | "
             f"{'🔴 free-ship dominant — fee ≠ cost' if sf['zero_pct']>50 else '🟡 partial free-ship'} |")
    a.append(f"| Mean fee | {sf['mean_fee']:,.0f} VND | — |")
    a.append(f"| Median fee | {sf['median_fee']:,.0f} VND | — |")
    a.append(f"| Total fee collected | {_fmt_vnd(sf['total_fee'])} | "
             f"= {sf['total_fee']/m['net_revenue']*100:.3f}% revenue |")
    a.append("")
    a.append("> **Verdict**: Field `shipments.shipping_fee` ghi *fee thu từ khách*, "
             "không phải *cost trả carrier*. Industry shipping cost benchmark "
             "8-12% revenue — chart 33 dùng sensitivity scenarios để correct cho "
             "data gap này.")
    a.append("")
    a.append("![Data Quality Audit](figures/00_data_quality.png)")
    a.append("")
    a.append("---")

    # ---------- 3. Pipeline Architecture ----------
    a.append("## 3. Pipeline Architecture")
    a.append("")
    a.append("### 3.1 Data Sources (13 CSV across 4 layers)")
    a.append("| Layer | Files |")
    a.append("|---|---|")
    a.append("| **Master** | `products`, `customers`, `promotions`, `geography` |")
    a.append("| **Transaction** | `orders`, `order_items`, `payments`, `shipments`, "
             "`returns`, `reviews` |")
    a.append("| **Analytical** | `sales` |")
    a.append("| **Operational** | `inventory`, `web_traffic` |")
    a.append("")
    a.append("### 3.2 Three Analytical Base Tables (ABTs)")
    a.append("- **`tx_master`** — order-line grain. Money features (`gross_revenue`, "
             "`net_revenue`, `cogs_total`, `gross_profit`, `margin_pct`, "
             "`discount_pct`), calendar, flags (`is_promo`, `is_full_price`, "
             "`is_returned`).")
    a.append("- **`customer_abt`** — customer grain. RFM quintile scoring, 6-segment "
             "assignment, CLV, tenure, cohort, AOV, `days_to_first_ord`, "
             "`full_price_share`.")
    a.append("- **`time_master`** — daily grain. Revenue, traffic, bounce, conv-rate, "
             "promo activity.")
    a.append("")
    a.append("### 3.3 Storytelling Arc — 6 Acts (+ Act 0 integrity)")
    a.append("| Act | Câu hỏi | Rubric level |")
    a.append("|---|---|---|")
    a.append("| 0. Data Quality | Data có đáng tin? | Foundation |")
    a.append("| 1. Business Pulse | What happened? | Descriptive |")
    a.append("| 2. Customers | Who / why? | Diagnostic |")
    a.append("| 3. Revenue drivers | Why / what next? | Diagnostic → Predictive |")
    a.append("| 4. Operational leaks | Where's the leak? | Diagnostic |")
    a.append("| 5. Strategic outlook | What's the move? | Predictive → Prescriptive |")
    a.append("| 6. Retail-grade metrics | Industry-benchmarked view | Cross-cutting |")
    a.append("")
    a.append("---")

    # ---------- 3.4 Critical Items Resolution — v6 Final ----------
    a.append("## 3.4 Critical Items Resolution — v6 Final")
    a.append("")
    a.append("v6 báo cáo phản hồi đầy đủ các issues từ review v3-v5, trong đó 3 "
             "fundamental fixes (signup integrity, scope reconciliation, contribution "
             "margin honesty) tái định hình toàn bộ narrative.")
    a.append("")
    a.append("| # | Issue (v3-v5) | v6 fix |")
    a.append("|---|---|---|")
    a.append("| 1 | Burying the lede — bombshell framing dùng CM −1.6% như fact, "
             "trong khi shipping cost không có trong dataset | "
             "**HEADLINE FINDINGS section ở đầu report** với sensitivity table; "
             "CM as-data = +2.24%, sensitivity range −9.74% to +2.24% across "
             "realistic shipping scenarios |")
    a.append("| 2 | Shipping cost 0.02% bất khả thi (vs benchmark 8-12%) | "
             "**Chart 33 sensitivity panel** + DQ Section 2.4 diagnose: dataset's "
             "`shipping_fee` là customer-paid fee, không phải carrier cost. "
             "Carrier cost không có trong dataset |")
    a.append("| 3 | Cancelled orders inflate revenue ~9% across all v3-v5 KPIs | "
             "**`[A3_realized_order_def]` enforcement** filter cancelled trong "
             "narrative; separate forecast scope keeps cancelled per BTC target "
             "definition |")
    a.append("| 4 | `signup_date` field unreliable (89% customers have signup "
             "AFTER first order) | **`activation_cohort` anchor** thay signup-based "
             "cohort heatmap; tenure rebuilt from first_order_date |")
    a.append("| 5 | Silent cohort 27.7% silently dropped from RFM | "
             "**Finding 3 + activation funnel** make silent cohort visible; "
             "true activation rate 72.3% reported |")
    a.append("| 6 | Predictive coverage chỉ 5/44 = 11% | **Chart 45** Probabilistic "
             "CLV + Churn distribution (Fader & Hardie 2005) |")
    a.append("| 7 | Prescriptive thiếu quantified tradeoffs | **Section 7.5 "
             "Value-at-Stake** — 7 recommendations với annual uplift, investment, "
             "payback months, risk |")
    a.append("| 8 | DQ Audit chỉ mô tả checks, không show results | **Section 2** "
             "với 4 bảng kết quả thực: nulls, integrity, date coverage, shipping "
             "diagnostic |")
    a.append("| 9 | Chỉ so industry avg, chưa so best-in-class | **Section 5.1** "
             "thêm 3-way: Industry · Inditex · Shein · ASOS |")
    a.append("| 10 | Inventory data tracks chỉ 28.8% transaction volume — relative "
             "rankings unreliable | **Caveat banner** ở đầu ACT 9 (Charts 17, 18, "
             "30, 37, 38, 43); insights re-framed as \"signals from subset\" |")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 3.5 Strategist feedback — v3 gaps addressed ----------
    a.append("## 3.5 Strategist Feedback — v3 Gaps Addressed")
    a.append("")
    a.append("Bản v3 này phản hồi trực tiếp 7 gap của một Fashion Retail Strategist "
             "senior (góc nhìn CFO / Head of Merchandising):")
    a.append("")
    a.append("| # | Gap | Chart phản hồi | Nguồn chuẩn |")
    a.append("|---|---|---|---|")
    a.append("| #1 | Unit Economics chỉ dừng ở Gross Margin | **Chart 33** "
             "Contribution Margin Waterfall (−shipping −payment −packaging "
             "−return processing) | Reichheld & Schefter HBR 2000 |")
    a.append("| #2 | Thiếu revenue decomposition Traffic × CVR × UPT × AUR | "
             "**Chart 34** 4-driver index chart | Levy & Weitz Ch.5 |")
    a.append("| #3 | Vietnam / SEA context vắng | **Chart 35** COD cancellation "
             "diagnostic + **Chart 36** VN calendar event lift (Tết, 11.11, 12.12, "
             "8/3, 20/10, Valentine) | Ninja Van 2023; Momentum Works 2024 |")
    a.append("| #4 | Customer behavior thiếu chiều sâu | **Chart 39** Category "
             "penetration matrix + First-purchase → LTV · **Chart 40** Wardrobing "
             "flag + Cancel vs Return split | Kumar & Shah 2004; Appriss 2023 |")
    a.append("| #5 | Thiếu merchandising KPI cốt lõi | **Chart 37** Full-Price "
             "STR @ Week 6 · **Chart 38** Aged inventory + Newness ratio · "
             "**Chart 41** Price–COGS paradox + Price ladder | "
             "Fisher & Raman HBR 1994; Inditex/Zara ops |")
    a.append("| #6 | ESG / sustainability hoàn toàn vắng | **Chart 43** "
             "Return carbon footprint + Deadstock risk | McKinsey State of Fashion "
             "2024; BoF Sustainability Index |")
    a.append("| #7 | Qualitative text signal chưa khai thác | **Chart 42** "
             "Review keywords by rating + NPS proxy | Ghose & Ipeirotis MIS "
             "Quarterly 2011; Reichheld HBR 2003 |")
    a.append("")
    a.append("**3 câu CFO/board sẽ hỏi — v6 trả lời trực tiếp:**")
    a.append("")
    a.append(f"1. *Contribution margin sau variable costs?* → "
             f"**{m['contribution_margin_pct']*100:+.2f}% goods-only** "
             f"({_fmt_vnd(m['contribution_margin'])} scaled), excluding shipping "
             f"per [A9_shipping_treatment] (carrier cost không có trong dataset). "
             f"Even at favourable +2% goods-only, true CM-with-logistics likely "
             f"thin or negative if logistics tracks industry 8-12% benchmark. "
             f"Chart 33 waterfall.")
    a.append(f"2. *COD cancel rate bao nhiêu? Có đốt logistics không?* → "
             f"COD chiếm **{m['cod_share']*100:.2f}%** orders (thấp xa so với "
             f"benchmark VN {BENCHMARKS['cod_share_vn']*100:.0f}%, nghi vấn dataset bias); "
             f"COD cancel rate **{m['cancel_rate_cod']*100:.2f}%** (dưới benchmark VN "
             f"{BENCHMARKS['cod_cancel_rate_vn']*100:.0f}%). Chart 35.")
    a.append(f"3. *Margin {m['gross_margin']*100:.2f}% với full-price share "
             f"{m['full_price_rev_share']*100:.1f}% — đâu là root cause?* → "
             f"**{m['sku_negative_margin_pct']*100:.1f}% SKUs** có cogs ≥ price "
             f"(list pricing không sai); median list margin "
             f"**{m['median_list_margin_pct']*100:.2f}%** healthy. Vấn đề là "
             f"realized margin sau cancellation filter + variable cost stack, "
             f"không phải pricing decision per-SKU. Chart 41 localize.")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 4. Fashion Pain Points → Solutions ----------
    a.append("## 4. Fashion E-commerce Pain Points Addressed")
    a.append("")
    a.append("Mỗi chart trong pipeline được design xoay quanh 1 trong 10 pain-point "
             "điển hình của fashion e-commerce (ref: McKinsey State of Fashion, "
             "BoF, Shopify, Narvar, Baymard).")
    a.append("")
    a.append("| ID | Pain point | Research fact | Solution lever |")
    a.append("|---|---|---|---|")
    for pid, p in PAIN_POINTS.items():
        a.append(f"| `{pid}` | **{p['name']}** | {p['fact']} | {p['lever']} |")
    a.append("")
    a.append("---")

    # ---------- 5. Industry Benchmarks ----------
    a.append("## 5. Industry Benchmarks Reference")
    a.append("")
    a.append("Các ngưỡng được nhúng vào chart annotations để diagnose nhanh.")
    a.append("")
    a.append("### 5.1 Competitive 3-way comparison (v4 — Bain/McKinsey style)")
    a.append("")
    a.append("Không chỉ so vs industry median — so cả best-in-class để thấy "
             "trần thực sự của ngành.")
    a.append("")
    a.append("| Metric | Our actual | Industry median | Inditex (Zara) | "
             "Shein | ASOS |")
    a.append("|---|---|---|---|---|---|")
    a.append(f"| Gross margin | **{m['gross_margin']*100:.1f}%** | "
             f"{COMPETITIVE_BM['gross_margin']['industry']*100:.0f}% | "
             f"{COMPETITIVE_BM['gross_margin']['inditex']*100:.0f}% | "
             f"{COMPETITIVE_BM['gross_margin']['shein']*100:.0f}% | "
             f"{COMPETITIVE_BM['gross_margin']['asos']*100:.0f}% |")
    a.append(f"| Return rate | **{m['return_rate']*100:.1f}%** | "
             f"{COMPETITIVE_BM['return_rate']['industry']*100:.0f}% | "
             f"{COMPETITIVE_BM['return_rate']['inditex']*100:.0f}% | "
             f"{COMPETITIVE_BM['return_rate']['shein']*100:.0f}% | "
             f"{COMPETITIVE_BM['return_rate']['asos']*100:.0f}% |")
    a.append("")
    a.append("> **Đọc bảng**: Gross margin của ta thấp xa cả Shein (40%, được "
             "biết là margin mỏng nhất ngành). Return rate thấp hơn cả Inditex "
             "— nhiều khả năng do data issue (pre-delivery cancel chưa vào "
             "returns table) hoặc friction return policy, KHÔNG phải quality "
             "vượt trội.")
    a.append("")
    a.append("### 5.2 Industry benchmark thresholds")
    a.append("")
    a.append("| Metric | Benchmark |")
    a.append("|---|---|")
    a.append(f"| Online fashion return rate (avg) | {BENCHMARKS['return_rate_online_fashion']:.0%} |")
    a.append(f"| Healthy return-rate target | {BENCHMARKS['return_rate_healthy_target']:.0%} |")
    a.append(f"| Gross margin — pure-play | {BENCHMARKS['gross_margin_fashion']:.0%} |")
    a.append(f"| Gross margin — marketplace | {BENCHMARKS['gross_margin_marketplace']:.0%} |")
    a.append(f"| 12-month repeat rate | {BENCHMARKS['repeat_rate_12m']:.0%} |")
    a.append(f"| Cart abandon rate | {BENCHMARKS['cart_abandon_rate']:.0%} |")
    a.append(f"| Sessions → order CVR | {BENCHMARKS['sessions_to_order_cvr']:.1%} |")
    a.append(f"| Full-price sell-through target | {BENCHMARKS['full_price_sellthrough']:.0%} |")
    a.append(f"| GMROI target | ≥ {BENCHMARKS['gmroi_target']} |")
    a.append(f"| Inventory turns / year | {BENCHMARKS['inventory_turns_fashion']} |")
    a.append(f"| LTV:CAC healthy | ≥ {BENCHMARKS['ltv_cac_ratio_healthy']}:1 |")
    a.append(f"| Size-curve norm | S 15 · M 35 · L 35 · XL 15 (%) |")
    a.append(f"| Product lifecycle — fast-fashion peak | "
             f"tuần {BENCHMARKS['product_lifecycle_weeks_fast_fashion']} |")
    a.append(f"| Product lifecycle — traditional peak | "
             f"tuần {BENCHMARKS['product_lifecycle_weeks_traditional']} |")
    a.append("")
    a.append("---")

    # ---------- 6. Chart Catalog ----------
    a.append("## 6. Chart Catalog (52 visualizations)")
    a.append("")
    a.append("Mỗi chart kèm: *sources* (dataset merge) · *pain-point addressed* · "
             "*industry benchmark* · *methodology reference* · *business meaning* · "
             "PNG render.")
    a.append("")

    acts_order = ["ACT 0 — Data Quality & Integrity",
                  "ACT 1 — Business Pulse",
                  "ACT 2 — Who Are Our Customers?",
                  "ACT 3 — What Drives Revenue?",
                  "ACT 4 — Where Are We Bleeding?",
                  "ACT 5 — What Should We Do?",
                  "ACT 6 — Industry-Grade Retail Metrics",
                  "ACT 7 — Unit Economics (v3)",
                  "ACT 8 — Vietnam Market Context (v3)",
                  "ACT 9 — Fashion Merchandising (v3)",
                  "ACT 10 — Customer Behavior Deep-Dive (v3)",
                  "ACT 11 — Text Mining & ESG (v3)",
                  "ACT 12 — Decision Support (v4)"]

    by_act = {}
    for cid, meta in CHART_META.items():
        by_act.setdefault(meta["act"], []).append((cid, meta))

    # Per-act caveat banners (emitted once at top of an act if present)
    ACT_BANNERS = {
        "ACT 9 — Fashion Merchandising (v3)": [
            "> ⚠ **INVENTORY DATA SCOPE WARNING** (applies to Charts 17, 18, 30, 37, 38, 43)",
            "> ",
            "> Per `[inventory_oi_reconciliation]` audit: `inventory.units_sold` tracks "
            "chỉ **~28.8% of `order_items` volume**. Possible explanations: (a) sampled "
            "SKU subset của simulator, (b) single-warehouse view, (c) \"delivered only\" "
            "definition khác với order_items scope.",
            "> ",
            "> **Implication for reading**:",
            "> - Absolute numbers (deadstock VND, GMROI ratios, inventory aging) là "
            "**scaled signals**, không phải facts about the whole business",
            "> - Relative rankings between categories CÓ THỂ bị sampling bias nếu coverage "
            "không uniform across categories — bias check pending; treat rankings là "
            "directional, magnitude indicative",
            "> - Mọi merchandising recommendation từ ACT 9 nên test lại với full-coverage "
            "inventory data trước khi commit operational changes",
            "> ",
            "> Pipeline xử lý: chart annotations giữ benchmarks (Fisher & Raman 60% STR "
            "threshold, Zara/Shein 75% newness benchmark) cho diagnostic value, nhưng "
            "absolute gap-to-benchmark phải đọc với caveat trên.",
            "",
            "---",
            "",
        ],
    }

    for act_name in acts_order:
        a.append(f"### {act_name}")
        a.append("")
        for line in ACT_BANNERS.get(act_name, []):
            a.append(line)
        for cid, meta in by_act.get(act_name, []):
            pain_id = meta.get("pain","—")
            pain_desc = (PAIN_POINTS.get(pain_id, {}).get("name", "—")
                         if pain_id != "—" else "—")
            a.append(f"#### {cid[:2]}. {meta['title']}")
            a.append("")
            a.append(f"- **Sources merged**: `{'`, `'.join(meta['sources'])}`")
            a.append(f"- **Pain addressed**: `{pain_id}` — {pain_desc}")
            a.append(f"- **Industry benchmark**: {meta.get('benchmark','—')}")
            a.append(f"- **Methodology**: {meta.get('reference','—')}")
            a.append(f"- **Business meaning**: {meta['meaning']}")
            if meta.get("caveat"):
                a.append(f"- **Caveat (v6)**: {meta['caveat']}")
            a.append("")
            a.append(f"![{meta['title']}](figures/{cid}.png)")
            a.append("")
        a.append("---")
        a.append("")

    # ---------- 7. Methodology & References ----------
    a.append("## 7. Methodology & References")
    a.append("")
    a.append("Các framework, mô hình và nghiên cứu làm xương sống cho pipeline:")
    a.append("")
    a.append("| Method / Framework | Reference |")
    a.append("|---|---|")
    for name, ref in METHODOLOGY_REFS:
        a.append(f"| **{name}** | {ref} |")
    a.append("")
    a.append("### Tư tưởng thiết kế")
    a.append("")
    a.append("1. **Data Quality first** — không một KPI nào được tính trước khi "
             "pipeline verify integrity (Sculley et al. 2015 về hidden tech debt).")
    a.append("2. **Every chart answers a business question** — theo 4 cấp độ của "
             "đề thi (Descriptive → Diagnostic → Predictive → Prescriptive).")
    a.append("3. **Benchmark-driven diagnosis** — mỗi chart so sánh actual vs industry "
             "norm để nhanh chóng xác định outlier/gap, thay vì chỉ vẽ số liệu trần.")
    a.append("4. **Research-referenced** — mỗi phương pháp truy nguyên tới paper "
             "nguồn hoặc playbook của tập đoàn lớn (Zalando, Inditex, Shein, "
             "Amazon, Meta).")
    a.append("5. **Fashion-specific lens** — KPI được chọn kỹ cho ngành fashion "
             "(GMROI, full-price sell-through, size curve, lifecycle S-curve, "
             "return root-cause), không phải KPI generic e-com.")
    a.append("")
    a.append("---")

    # ---------- 7.5 Value-at-Stake recommendations (v4) ----------
    a.append("## 7.5 Value-at-Stake — Quantified Recommendations (v4)")
    a.append("")
    a.append("McKinsey Value-at-Stake methodology: mỗi recommendation phải có "
             "$ uplift, $ investment, payback months, risk rating. Tất cả tính "
             "trực tiếp từ data — không phải opinion.")
    a.append("")
    a.append("| Action | Annual uplift (M VND) | Investment (M VND) | "
             "Payback (months) | Owner | Risk |")
    a.append("|---|---|---|---|---|---|")
    vas = m.get("value_at_stake", [])
    for r in sorted(vas, key=lambda x: -x.get("annual_uplift_M", 0)):
        a.append(f"| {r['action']} | "
                 f"{r['annual_uplift_M']:,.0f} | "
                 f"{r['investment_M']:,.0f} | "
                 f"{r['payback_months']:.1f} | "
                 f"{r['owner']} | {r['risk']} |")
    if vas:
        total = sum(r["annual_uplift_M"] for r in vas)
        a.append(f"| **TOTAL** | **{total:,.0f}** | — | — | — | — |")
    a.append("")
    a.append("![Value-at-Stake](figures/44_value_at_stake.png)")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 7.6 Probabilistic CLV (v4) ----------
    a.append("## 7.6 Probabilistic CLV + Churn Distribution (v4)")
    a.append("")
    a.append("Simplified BG/NBD-flavoured CLV: P(alive) decays exponentially "
             "với recency, scaled bởi customer-specific avg gap. Xuất ra "
             "expected revenue 12 tháng tới per customer + churn-score "
             "distribution → input cho retention budget allocation.")
    a.append("")
    a.append("![CLV + Churn](figures/45_clv_churn.png)")
    a.append("")
    a.append("---")
    a.append("")

    # ---------- 8. Upgrade Roadmap ----------
    a.append("## 8. Upgrade Roadmap — vượt rubric Prescriptive")
    a.append("")
    a.append("### 8.1 Causal inference (gold-standard)")
    a.append("- **Double ML / Propensity Score** (Chernozhukov et al. 2018) cho "
             "promo lift — loại bias selection mà DiD proxy không xử lý được.")
    a.append("- **Uplift modeling** (2-model learners hoặc X-Learner, Künzel 2019) "
             "để tối ưu targeting: ai trong số at-risk thực sự *respond* với promo.")
    a.append("- **Synthetic control** cho region expansion test: vùng mới vs vùng "
             "tương tự chưa mở rộng.")
    a.append("")
    a.append("### 8.2 Probabilistic customer models")
    a.append("- **BG/NBD + Gamma-Gamma** (Fader & Hardie) — dự báo xác suất còn "
             "active + expected revenue cho từng khách 6-12 tháng tới.")
    a.append("- **Markov RFM transition matrix** — mô hình di chuyển giữa segments "
             "theo quý, validate retention playbook.")
    a.append("- **Hierarchical Bayesian CLV** (Stan/PyMC) khi data ít/noisy cho "
             "cohort mới.")
    a.append("")
    a.append("### 8.3 Advanced forecasting")
    a.append("- **LightGBM với lag × Fourier features + exogenous** (promo dummies, "
             "traffic, bounce) — leaderboard-proven baseline.")
    a.append("- **Temporal Fusion Transformer** (Lim et al. 2021 Google) — state-of-"
             "the-art interpretable deep forecasting.")
    a.append("- **Prophet + custom Vietnamese calendar** (Tết, Quốc Khánh, "
             "11.11, 12.12, Valentine, 8/3, 20/10).")
    a.append("- **Change-point detection** (`ruptures` hoặc Bayesian online CPD) — "
             "auto-annotate regime shifts cho narrative.")
    a.append("")
    a.append("### 8.4 Merchandising intelligence")
    a.append("- **Assortment optimization** knapsack: constraint = stockout-days, "
             "objective = gross profit → 'keep / cut' list.")
    a.append("- **Dynamic pricing** sensitivity model per SKU × region × week.")
    a.append("- **Buying depth recommendation**: từ lifecycle S-curve, predict "
             "optimal first-order qty cho SKU mới.")
    a.append("- **Markdown-cadence optimizer** — Marshall-inspired: khi nào giảm giá "
             "bao nhiêu để maximize gross profit, không chỉ sell-through.")
    a.append("")
    a.append("### 8.5 Deep personalization & content")
    a.append("- **Two-tower embedding** (Google YouTube-style) cho recommendation.")
    a.append("- **NLP review topic modeling** (BERTopic) trên `review_title` để "
             "auto-extract fit/quality/shipping issues.")
    a.append("- **Complete-the-look bundles** via association rules với minimum "
             "confidence + lift.")
    a.append("")
    a.append("### 8.6 Geospatial")
    a.append("- Choropleth cấp tỉnh VN với `geopandas` + `plotly.choropleth_mapbox`.")
    a.append("- **Spatial autocorrelation** (Moran's I) để kiểm tra cluster-vs-"
             "random pattern.")
    a.append("- **Service area modelling** — vẽ isochrone từ warehouse để tối ưu "
             "fulfilment network.")
    a.append("")
    a.append("### 8.7 Risk & governance")
    a.append("- **Customer concentration stress-test**: nếu top-1% customer churn, "
             "revenue giảm bao nhiêu?")
    a.append("- **HHI (Herfindahl-Hirschman)** cho supplier/category concentration.")
    a.append("- **Black-swan scenario modelling** — COVID-like shock replayed.")
    a.append("")
    a.append("### 8.8 Engineering productionization")
    a.append("- Chuyển pipeline sang **Polars / DuckDB** (≥10× faster trên data "
             "lớn).")
    a.append("- **dbt-style SQL models** + tests thay pandas merge — tái lập được.")
    a.append("- **Dagster / Airflow** orchestration; **MLflow** experiment "
             "tracking; **Great Expectations** cho data contracts.")
    a.append("- **Quarto / Marimo** notebook → MD + PDF + HTML từ 1 nguồn.")
    a.append("")
    a.append("---")
    a.append("")
    a.append("*Generated by `datathon_2026_eda.py` v2 — VinTelligence Datathon 2026*")

    out_path.write_text("\n".join(a), encoding="utf-8")
    print(f"\n✅ Markdown report written to {out_path}")


# =============================================================================
# 17. MAIN
# =============================================================================
def main():
    print("="*70); print("LOADING DATA"); print("="*70)
    d = load_raw()

    print("\n" + "="*70); print("DATA QUALITY AUDIT (ACT 0)"); print("="*70)
    chart_00_data_quality(d)

    print("\n" + "="*70); print("BUILDING ANALYTICAL BASE TABLES"); print("="*70)
    tx   = build_tx_master(d)
    cust = build_customer_master(d, tx)
    tm   = build_time_master(d, tx)
    silent = build_silent_cohort(d, tx)            # Per [silent_cohort]
    print(f"  tx_master    : {tx.shape}")
    print(f"  customer_abt : {cust.shape} (active only — Per [A4_active_customer_def])")
    print(f"  silent_cohort: {silent.shape} (Per [silent_cohort])")
    print(f"  time_master  : {tm.shape}")
    print(f"  cancelled rows in tx (kept for flow analysis, filtered for revenue): "
          f"{(tx['is_cancelled']==1).sum():,}")

    print("\n" + "="*70); print("COMPUTING METRICS"); print("="*70)
    metrics = compute_metrics(tx, cust, tm, d)
    print(f"  Net revenue       : {metrics['net_revenue']:,.0f}")
    print(f"  Gross margin      : {metrics['gross_margin']*100:.1f}%  "
          f"(vs industry {BENCHMARKS['gross_margin_fashion']*100:.0f}%, "
          f"gap {metrics['vs_margin_bm']*100:+.1f} pp)")
    print(f"  Return rate       : {metrics['return_rate']*100:.2f}%  "
          f"(vs industry {BENCHMARKS['return_rate_online_fashion']*100:.0f}%)")
    print(f"  Repeat rate       : {metrics['repeat_rate']*100:.2f}%  "
          f"(vs industry {BENCHMARKS['repeat_rate_12m']*100:.0f}%)")
    print(f"  SKU Gini          : {metrics['sku_gini']:.3f}")
    print(f"  Top-10% cust rev  : {metrics['top10pct_cust_rev_share']*100:.1f}%")
    print(f"  Contribution mgn  : {metrics['contribution_margin_pct']*100:+.2f}%  "
          f"(after ship + payment + packaging + returns)")
    print(f"  COD share         : {metrics['cod_share']*100:.1f}%  "
          f"· COD cancel rate {metrics['cancel_rate_cod']*100:.2f}%")
    print(f"  Neg-margin SKUs   : {metrics['sku_negative_margin_pct']*100:.1f}%")
    print(f"\n  🚨 HEADLINE — Contribution Margin (goods-only per A9):")
    print(f"     - CM goods-only  : {metrics['contribution_margin_pct']*100:+.2f}%")
    print(f"     - Memo shipping  : {metrics['shipping_fee_pct_revenue']*100:.3f}% of revenue "
          f"(customer-paid fee; NOT in CM per A9_shipping_treatment)")

    print("\n" + "="*70); print("GENERATING CHARTS (25 core) → PNG"); print("="*70)
    chart_01_kpi_scorecard(tx, cust, tm)
    chart_02_revenue_waterfall(tx)
    chart_03_revenue_timeline_stl(tm, d["promotions"])
    chart_04_yoy_heatmap(tm)
    chart_05_dow_month_heatmap(tm)
    chart_06_rfm_scatter(cust)
    chart_07_cohort_heatmap(cust, tx)
    chart_08_clv_by_channel(cust)
    chart_09_demographic_sunburst(tx)
    chart_10_geo_map(tx)
    chart_11_bcg_matrix(tx)
    chart_12_price_elasticity(tx)
    chart_13_promo_roi_bubble(tx, d["promotions"])
    chart_14_size_color_returns(tx)
    chart_15_copurchase_network(tx)
    chart_16_funnel_by_source(tm, d, tx)
    chart_17_inventory_quadrant(d, tx)
    chart_18_dos_vs_sellthrough(d)
    chart_19_returns_sankey(d, tx)
    chart_20_shipping_sla(d, tx)
    chart_21_review_vs_return(d, tx)
    chart_22_forecast_preview(tm)
    chart_23_churn_clv_matrix(cust)
    chart_24_traffic_revenue_lag(tm)
    chart_25_action_priority()

    print("\n" + "="*70); print("GENERATING RETAIL-GRADE CHARTS (7 new)"); print("="*70)
    chart_26_sku_pareto(tx)
    chart_27_survival_curves(cust, tx)
    chart_28_size_curve(tx)
    chart_29_lifecycle_scurve(tx)
    chart_30_gmroi(tx, d)
    chart_31_promo_incrementality(tx)
    chart_32_anomaly_stl(tm, d["promotions"])

    print("\n" + "="*70)
    print("GENERATING v3 STRATEGIST-GAP CHARTS (11 new, charts 33–43)")
    print("="*70)
    chart_33_contribution_margin(tx, d)
    chart_34_revenue_decomposition(tm, tx)
    chart_35_cod_cancellation(d, tx)
    chart_36_vn_calendar_lift(tm)
    chart_37_str_week6(tx, d)
    chart_38_aged_newness(tx, d)
    chart_39_category_penetration(cust, tx)
    chart_40_wardrobing(tx, d)
    chart_41_price_paradox(tx, d)
    chart_42_review_nps(d)
    chart_43_esg_risks(tx, d)

    print("\n" + "="*70)
    print("GENERATING v4 DECISION-SUPPORT + v6 ACTIVATION FUNNEL + v6 DIAGNOSTICS (44–49)")
    print("="*70)
    _, vas_df = chart_44_value_at_stake(tx, cust, d, metrics)
    _, _      = chart_45_clv_churn(cust, tx)
    chart_46_activation_funnel(d)                   # Per Finding 3 + [silent_cohort]
    chart_47_signup_gap_distribution(d)             # Diagnostic A
    chart_48_cancelled_only_profile(d)              # Diagnostic B
    chart_49_activation_clv_overlay(d)              # Diagnostic C
    chart_50_phantom_catalog_composition(d, tx)     # v6 follow-up: phantom catalog
    _, scen_df = chart_51_scenario_projection(tx, d, metrics)  # v6 follow-up: scenarios
    metrics["value_at_stake"] = vas_df.to_dict(orient="records")
    metrics["scenario_projection"] = scen_df.to_dict(orient="records")

    print("\n" + "="*70); print("WRITING MARKDOWN REPORT"); print("="*70)
    build_markdown_report(metrics, OUTPUT_DIR / "EDA_REPORT.md")

    tx.to_parquet(OUTPUT_DIR / "tx_master.parquet", index=False)
    cust.to_parquet(OUTPUT_DIR / "customer_abt.parquet", index=False)
    tm.to_parquet(OUTPUT_DIR / "time_master.parquet", index=False)
    silent.to_parquet(OUTPUT_DIR / "silent_cohort.parquet", index=False)
    print(f"\n ABTs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
