"""Simple inference script using a trained classifier"""

import pandas as pd
import torch
from transformers import AutoTokenizer

from model import LMClassifier

# Predefined labels
labels = [
    "Absence of significant revenues",
    "Accumulated/retained earnings deficit",
    "Act of God (Extreme weather, War, Illness, Even Death, etc)",
    "Assets û inadequate, limited, immaterial or impaired",
    "Bankruptcy",
    "Benefit Plan, Pension, etc. - Obligations",
    "Changed industry or business",
    "Compensation deferred",
    "Credit line reduced, unavailable or due",
    "Credit quality deterioration",
    "Debt covenants/agreements uncertain or not in compliance",
    "Debt is substantial",
    "Decline in revenue",
    "Derivatives - obligations, losses",
    "Development stage",
    "Discontinued/Disposal of Operations",
    "Exploration/Pre-exploration Stage",
    "Gross margin û negative",
    "Initial loss",
    "Insufficient / limited cash, capital or liquidity concerns",
    "Liabilities exceed assets",
    "Limited Performance/Credit History",
    "Liquidation of assets or divestitures",
    "Litigation contingencies",
    "Need for additional financing for funding obligations and/or servicing debt",
    "Need for additional financing for growth or to meet business objectives",
    "Need for additional financing to sustain operations",
    "Negative cash flow from operations",
    "Net capital deficiency",
    "Net losses since inception",
    "Net/Operating Loss (including recurring losses)",
    "No Marketable Product(s)",
    "No dividends",
    "No explanation",
    "Not commenced, limited or no operations",
    "Notes Payable/Debt Maturity; Balance Due, Past-due, Default",
    "Pending Dissolution/Contract Expiration or Termination",
    "Product demand or pricing - decline or limited",
    "Profitability concerns",
    "Recoverability of (natural) resources - uncertain",
    "Refinancing contingencies",
    "Regulatory capital - decline or deficiency",
    "Regulatory settlements, obligations and contingencies",
    "Related Party/Segment Issues",
    "Restructuring contingencies",
    "Reverse merger",
    "Seeking or needs to combine with existing company",
    "Significant contractual obligations & commitments pending",
    "Significant expenses",
    "Stock/share Redemption or Option Exercise Risk(s)",
    "Stockholder equity or partner capital - deficiency or decrease",
    "Subsidiary - spin off",
    "Tax liability-deferred, disputed, unfunded",
    "Vendor-supplier disputes or disruptions",
    "Working capital/current ratio deficit/inadequacy",
]


# Load the model
model_path = "./model.pt"
model = LMClassifier(
    num_output=len(labels),
)

device = "cpu"

model.to(device)
print(model.load_state_dict(torch.load(model_path)))

# Sample text (use your own)

text = "Report of Independent Registered Accounting Firm  To the Stockholders and Board of Directors  Lordstown Motors Corp.:  Opinion on the Consolidated Financial Statements  We have audited the accompanying consolidated balance sheets of Lordstown Motors Corp. and subsidiary (the Company) as of December 31, 2020 and the related consolidated statements of operations, stockholders equity, and cash flows for the year ended December 31, 2020, and the related notes (collectively, the consolidated financial statements). In our opinion, the consolidated financial statements present fairly, in all material respects, the financial position of the Company as of December 31, 2020, and the results of its operations and its cash flows for the year ended December 31, 2020, in conformity with U.S. generally accepted accounting principles.  Correction of Misstatements  As discussed in Note 2 to the consolidated financial statements, the 2020 financial statements have been restated to correct certain misstatements.  Going Concern  The accompanying consolidated financial statements have been prepared assuming that the Company will continue as a going concern. As discussed in Note 1, the Company does not have sufficient liquidity to fund commercial scale production and the launch of sale of its electric vehicles which raises substantial doubt about the Companys ability to continue as a going concern. Managements plans in regard to these matters are also described in Note 1. The consolidated financial statements do not include any adjustments that might result from the outcome of this uncertainty.  Basis for Opinion  These consolidated financial statements are the responsibility of the Companys management. Our responsibility is to express an opinion on these consolidated financial statements based on our audit. We are a public accounting firm registered with the Public Company Accounting Oversight Board (United States) (PCAOB) and are required to be independent with respect to the Company in accordance with the U.S. federal securities laws and the applicable rules and regulations of the Securities and Exchange Commission and the PCAOB.  We conducted our audit in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit to obtain reasonable assurance about whether the consolidated financial statements are free of material misstatement, whether due to error or fraud. Our audit included performing procedures to assess the risks of material misstatement of the consolidated financial statements, whether due to error or fraud, and performing procedures that respond to those risks. Such procedures included examining, on a test basis, evidence regarding the amounts and disclosures in the consolidated financial statements. Our audit also included evaluating the accounting principles used and significant estimates made by management, as well as evaluating the overall presentation of the consolidated financial statements. We believe that our audit provides a reasonable basis for our opinion.  /s/ KPMG LLP  We have served as the Companys auditor since 2020.  New York, New York  March 24, 2021, except for Notes 1, 2, 3, 4, and 13 as to which the date is June 8, 2021"

# For the sample text provided the correct labels are:
# [
#     "Insufficient / limited cash, capital or liquidity concerns",
#     "No Marketable Product(s)",
# ]


# Load tokenizer and parse input
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
)

# Run inference
model.eval()
with torch.no_grad():
    res = model(**data)
probs = (torch.nn.Sigmoid()(res)).numpy().flatten().tolist()


# Print results
df = pd.DataFrame({"Going Concern": labels, "Probability": probs})
df.sort_values("Probability", ascending=False, inplace=True)
print(df.to_string())
