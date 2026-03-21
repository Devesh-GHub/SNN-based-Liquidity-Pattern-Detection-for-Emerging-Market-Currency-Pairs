# Project Motivation — SNN-Based Direct INR/BRL Settlement System

**Date:** Day 14  

---

## The Problem

Indian exporters transacting with Brazilian importers currently lose over 
4% of transaction value — approximately ₹42,000 on every ₹10 lakh payment 
— due to mandatory USD routing through correspondent banks (SBI/HDFC fee 
research, Day 13). Settlement takes T+2 to T+3 business days, locking up 
working capital that export-dependent SMEs cannot afford to idle. No direct 
INR/BRL market exists; the rate is always derived through USD, embedding 
double conversion costs that neither party can negotiate away.

## Why Existing Solutions Fail

Traditional ML approaches (LSTM, Transformer) applied to FX prediction 
consume continuous compute proportional to time-series length, making 
real-time settlement decisions energetically and computationally expensive 
at scale. More critically, current proposed solutions — including the BIS 
mBridge project and RBI's e₹-W pilot — focus entirely on settlement 
*infrastructure* (ledgers, nodes, consensus protocols) and provide no 
mechanism for predicting optimal settlement windows or detecting FX 
anomalies before they propagate into settlement costs.

## Why This Solution

Spiking Neural Networks process information only when a meaningful signal 
occurs — matching the natural "burstiness" of FX return distributions, 
where most periods are quiet and rare events carry most of the information. 
This sparse, event-driven computation makes SNNs more energy-efficient than 
dense ANNs for financial time-series and provides a natural fit for encoding 
FX rate movements as spike trains, where a spike represents a settlement-relevant 
price event rather than continuous background noise.

## Policy Alignment

The RBI's 2022 CBDC Concept Note explicitly identifies cross-border payment 
innovation as a primary motivation for the Digital Rupee (e₹-W), stating 
that future pilots will focus on cross-border payments and settlement 
efficiency. The BIS mBridge project — of which the Reserve Bank of India 
is an observer member — demonstrates that multi-CBDC direct settlement is 
technically and regulatorily feasible; this project provides the missing 
analytical intelligence layer that neither the RBI's CBDC roadmap nor 
mBridge currently addresses.

## The Customer

*"As an Indian textile exporter in Surat who invoices a São Paulo importer 
in BRL, I currently lose ₹42,080 on every ₹10,00,000 transaction — over 
4% of my invoice value — due to the mandatory INR→USD→BRL routing through 
correspondent banks. I also wait 2–3 business days for settlement, tying 
up working capital I need to fund the next production run.*

*With a direct INR/BRL settlement system powered by SNN-based liquidity 
prediction, I would pay approximately ₹1,180 per ₹10 lakh transaction — 
saving ₹40,900 per trade. At 5 transactions per month, that is ₹2,04,500 
saved annually. Settlement would occur at T+0, eliminating the working 
capital gap entirely."*

---

*Sources: RBI CBDC Concept Note (October 2022); BIS mBridge Project Report 
(October 2022, June 2024 MVP); SBI/HDFC fee research (Day 13 primary research).*