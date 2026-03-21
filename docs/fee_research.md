# Fee Research Report — BRICS SNN FX Settlement Project

**Purpose:** Establish the real-world cost baseline that the SNN settlement 
system will be benchmarked against. This document becomes the business 
justification section of the thesis.

---

## 1. SBI Outward Remittance Fees (India → Brazil)

**Source:** SBI FxOut official page (retail.sbi.bank.in/sbijava/retail/html/fxout_FAQ.html)  
**Accessed:** [today's date]

### Flat transaction fees (SBI FxOut service)
| Amount | SBI Fee |
|---|---|
| Any outward remittance | ₹500 – ₹2,000 depending on method |
| Rupee outward (INR to INR-accepting countries) | 0.125% of amount, min ₹125 |

**Important:** Brazil does NOT accept INR directly. An INR → BRL transfer 
must route through USD. SBI's FxOut service covers USD, EUR, GBP, SGD, 
AUD, CAD, NZD, AED — not BRL. This forces a two-leg conversion:  
**INR → USD (SBI FxOut) → BRL (Brazilian correspondent bank)**

### Exchange rate markup (TT Selling Rate)
- SBI applies its **TT Selling Rate** to the mid-market rate
- Typical markup: **1.0% to 3.0%** above the interbank mid-market rate
- This markup is NOT disclosed as a line item — it is embedded in the rate
- On a ₹10,00,000 transfer, a 2% markup = **₹20,000 hidden cost**

### GST on forex conversion
- Indian government levies 18% GST on the bank's service fee component
- GST is charged on the flat fee, not the total transfer value

---

## 2. HDFC Bank Outward Remittance Fees

**Source:** HDFC Bank official fee page (hdfcbank.com/personal/pay/money-transfer/remittance/fees-and-charges)  
**Accessed:** [today's date]

### Flat transaction fees
| Amount | HDFC Fee |
|---|---|
| Up to USD 500 equivalent | ₹500 + applicable GST |
| Above USD 500 equivalent | ₹1,000 + applicable GST |

### Exchange rate markup (TT Selling Rate)
- HDFC markup on TT Selling Rate: **2.0% to 3.5%** above mid-market
- On a ₹10,00,000 transfer at 3% markup = **₹30,000 hidden in the rate**

### Additional charges
- Correspondent bank deductions: **$15–$50** (₹1,250–₹4,150 at ₹83/USD)
  deducted in-transit, before the recipient's bank even receives funds
- Brazilian receiving bank may deduct additional IOF tax (Brazil's financial 
  operations tax): typically **0.38%** on foreign currency inflows

---

## 3. SWIFT Transfer Timeline: India to Brazil

**Sources:**  
- Wise Help Centre: "Swift payments usually take 1–6 working days" , however it 
  is possible that they can take longer due to time differences between 
  the sending and receiving country
  (wise.com/help/articles/2553074/paying-by-swift)  
- SBI Karbon: "Outward transfers 2–7 business days depending on destination"  
  (karboncard.com/blog/sbi-international-money-transfer-charges)  
- ExiapIndia: SBI SWIFT payments "3–5 working days"  
  (exiap.com.in/guides/sbi-international-money-transfer)

### Why India → Brazil is T+2 to T+3 minimum

The INR → BRL route has **no direct SWIFT corridor**. The payment chain is:
```
Indian Bank (INR)
    ↓  [SBI FxOut / HDFC SWIFT — Day 0]
US Correspondent Bank (USD)
    ↓  [Overnight processing — Day 1]
Brazilian Correspondent Bank (USD → BRL conversion)
    ↓  [Local Brazilian clearing — Day 2–3]
Brazilian Beneficiary Bank (BRL)
```

Each hop adds 1 business day minimum. Brazil's local settlement 
system (STR — Sistema de Transferência de Reservas) adds a further 
local clearing delay. **Realistic settlement: T+2 to T+3 business days.**

For a textile exporter awaiting payment to fund next production cycle, 
this is 2–3 days of working capital locked up per transaction.

---

## 4. The Cost Formula

### Transaction size: ₹10,00,000 (10 lakh INR)
*Rationale: realistic for a mid-sized Indian textile exporter receiving 
a payment from a Brazilian importer. Equivalent to approximately $12,000 
USD or ~BRL 60,000 at current rates.*

---

### CURRENT ROUTE (INR → USD → BRL via SWIFT)

| Cost Component | Basis | Amount (₹) |
|---|---|---|
| SBI/HDFC flat remittance fee | ₹1,000 (above USD 500) | ₹1,000 |
| GST on flat fee (18%) | 18% × ₹1,000 | ₹180 |
| INR→USD TT spread (2% mid-point estimate) | 2% × ₹10,00,000 | ₹20,000 |
| US correspondent bank fee | $25 × ₹84/USD | ₹2,100 |
| USD→BRL spread at Brazilian bank (1.5% est.) | 1.5% × ₹10,00,000 | ₹15,000 |
| Brazilian IOF tax (0.38%) | 0.38% × ₹10,00,000 | ₹3,800 |
| **TOTAL CURRENT COST** | | **₹42,080** |
| **Effective cost rate** | | **4.21%** |               :- Effective Cost Rate = ( Total Cost / Amount Sent ) × 100 --> (42,080 / 10,00,000) x 100
| **Settlement time** | | **T+2 to T+3** |

*Sources: SBI/HDFC fee pages, SWIFT correspondent fee ranges from 
karboncard.com, IOF rate from Brazilian central bank standard schedule.*

---

### PROPOSED ROUTE (Direct INR/BRL via SNN-powered settlement)

| Cost Component | Basis | Amount (₹) |
|---|---|---|
| Direct conversion fee (assumed) | 0.1% × ₹10,00,000 | ₹1,000 |
| No correspondent banks | 0 hops | ₹0 |
| No double FX conversion | Direct pair | ₹0 |
| GST on service fee (18%) | 18% × ₹1,000 | ₹180 |
| **TOTAL PROPOSED COST** | | **₹1,180** |
| **Effective cost rate** | | **0.12%** |
| **Settlement time** | | **T+0 (near real-time)** |

**⚠️ ASSUMPTION NOTE:** The 0.1% fee for the proposed route is an 
assumption based on blockchain/DLT settlement benchmarks from academic 
literature (Bank for International Settlements, 2022). It is explicitly 
labelled as an assumption throughout the thesis. It is conservative 
relative to some proposals (which suggest <0.05%) and deliberately 
chosen to avoid overstating savings.

---

### SAVINGS CALCULATION

| Metric | Current | Proposed | Saving |
|---|---|---|---|
| Cost per ₹10L transaction | ₹42,080 | ₹1,180 | **₹40,900** |
| Effective cost rate | 4.21% | 0.12% | **4.09 percentage points** |
| Cost reduction | — | — | **~97.2%** |
| Settlement time | T+2 to T+3 | T+0 | **2–3 days faster** |

**This is your "X%" headline figure: ~97% cost reduction.**

For the thesis, this will be stated conservatively as:  
*"The proposed system reduces per-transaction cost by approximately 90–97%, 
with the lower bound reflecting a more conservative 0.2% proposed fee."*

---

## 5. User Story

> *"As an Indian textile exporter in Surat who invoices a São Paulo 
> importer in BRL, I currently lose ₹42,080 on every ₹10,00,000 
> transaction — over 4% of my invoice value — due to the mandatory 
> INR→USD→BRL routing through correspondent banks. I also wait 2–3 
> business days for settlement, tying up working capital I need to 
> fund the next production run.*
>
> *With a direct INR/BRL settlement system powered by SNN-based 
> liquidity prediction, I would pay approximately ₹1,180 per ₹10 
> lakh transaction — a saving of ₹40,900 per trade. At 5 transactions 
> per month, that is ₹2,04,500 saved annually — enough to fund one 
> additional loom operator's annual salary.*
>
> *Settlement would occur in T+0 instead of T+2, eliminating the 
> working capital gap entirely."*

---

## 6. Sources (with URLs and access dates)

| # | Source | URL | Date Accessed |
|---|---|---|---|
| 1 | SBI FxOut FAQ (official) | retail.sbi.bank.in/sbijava/retail/html/fxout_FAQ.html | [12-03-2026] |
| 2 | HDFC Remittance Fees (official) | hdfcbank.com/personal/pay/money-transfer/remittance/fees-and-charges | [12-03-2026] |
| 3 | SBI charges breakdown (Karbon) | karboncard.com/blog/sbi-international-money-transfer-charges | [12-03-2026] |
| 4 | HDFC charges breakdown (moneyHOP) | moneyhop.co/blog/hdfc-charges-for-international-transactions | [12-03-2026] |
| 5 | SWIFT timing (Wise Help) | wise.com/help/articles/2553074/paying-by-swift | [12-03-2026] |
| 6 | Correspondent bank fees (RemitBee) | remitbee.com/blog/finances/manage-money/bank-charges-for-receiving-international-transfers-in-india | [12-03-2026] |
| 7 | Hidden FX charges explained (Skydo) | skydo.com/blog/avoid-hidden-international-transaction-charges | [12-03-2026] |

---

## 7. Notes for Thesis Writing

- **Never claim** the savings are guaranteed — always write  
  "under the modelled assumptions"
- **Always cite** the 0.1% proposed fee as an assumption  
  with a reference to BIS settlement cost literature
- The IOF tax (0.38%) is a real Brazilian tax — cite  
  Brazil's central bank (bcb.gov.br) when writing the thesis
- The "₹40,900 saving" figure is the headline for the dashboard.  
  The model's job is to validate the liquidity conditions under  
  which direct settlement is feasible — not to eliminate the fee  
  difference (that is a policy/infrastructure claim, not an ML claim).