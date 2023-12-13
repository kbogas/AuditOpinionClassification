import json

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoTokenizer

from model import LMClassifier

####################################### SETUP #################################


def predict_labels_from_text(text: str) -> list:
    """Helper function to generate predictions from a model

    Args:
        text (str): The audit opinion text

    Returns:
        list: the list of probabilites for each pre-defined label
    """

    # Parse the text
    data = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )
    data.to(device)
    # Run inference
    model.eval()
    with torch.no_grad():
        res = model(**data)
    probs = (torch.nn.Sigmoid()(res)).numpy().flatten().tolist()
    return probs


torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


# Predefined labels
labels = [
    "Absence of significant revenues",
    "Accumulated/retained earnings deficit",
    "Act of God (Extreme weather, War, Illness, Even Death, etc)",
    "Assets inadequate, limited, immaterial or impaired",
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
    "Gross margin negative",
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
device = "cpu"
model_path = "./model.pt"

model = LMClassifier(
    num_output=len(labels),
)

print(model.load_state_dict(torch.load(model_path, map_location=torch.device(device))))
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.to(device)

# Load the samples
with open("./samples.json", "r") as f:
    sample_texts = json.load(f)

############################# STREAMLIT APP ################################

# Set page title and favicon
st.set_page_config(
    page_title="📝 Audit Opinion Going Concern Labelling", page_icon="📝", layout="wide"
)

# First column with project information
st.sidebar.title("")
st.sidebar.image("app_logo.png")
st.sidebar.write(
    "Welcome to the **Audit Opinion Going Concern Labelling** (AOGCL) demo! This app uses a pre-trained LM to automatically predict Going Concern Issues based on auditor opinions."
)
st.sidebar.write(
    "Explore the details of the project, check out the code on GitHub, and cite the paper if applicable."
)
st.sidebar.markdown(
    "[![GitHub Repository](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/kbogas/AuditOpinionClassification)"
)
# "[GitHub Repository](https://github.com/kbogas/AuditOpinionClassification)"


st.sidebar.markdown("To Cite:")
st.sidebar.write(
    'Konstantinos Bougiatiotis, Elias Zavitsanos, Georgios Paliouras. *"Identifying going concern issues in auditor opinions: link to bankruptcy events"*. In proceedings of the 5th Financial Narrative Processing Workshop (FNP 2023) at the 2023 IEEE International Conference on Big Data (IEEE BigData 2023), Sorrento, Italy.'
)

# Main content area
st.title("🚀 Audit Opinion Going Concern Labelling (AOGCL) Demo")
st.write("Explore the power of automatic labelling in predicting Going Concern Issues!")

st.write("### 📚 Instructions:")
st.write(
    "1. Select a sample text from the drop-down menu or enter your own sample auditor opinion."
)
st.write("2. Click the 'Predict' button to see the model's predictions.")

# Create a sample text dropdown
selected_sample_text = st.selectbox(
    "📋 Select Sample Audit Opinion (Company with issues):",
    list(sample_texts.keys()),
    index=None,
)

# Create a text area for user input
user_input = st.text_area(
    "✏️ Or manually enter your text here:",
    value=sample_texts.get(selected_sample_text, ""),
    height=400,
)


# Set threshold for highlighting labels
threshold = 0.5

# Create a button to run the prediction
if st.button("Predict"):
    # Perform model inference on the user input
    probas = predict_labels_from_text(user_input)  # Replace with your actual function

    # Display the predicted probabilities with Plotly
    plot_data = pd.DataFrame(
        {
            "Label": labels,
            "Probability": probas,
        }
    )
    plot_data["Probability"] = 100 * plot_data["Probability"]

    # Sort the dataframe by probability in descending order
    plot_data = plot_data.sort_values(by="Probability", ascending=False)

    # Display the labels with probability > 0.5 as a list with probabilities
    high_prob_labels = plot_data[plot_data["Probability"] > threshold * 100]
    if len(high_prob_labels) > 0:
        st.write("### Labels with Probability > 0.5:")
        st.dataframe(high_prob_labels)

    # Create a Plotly bar plot
    fig = px.bar(
        plot_data,
        x="Label",
        y="Probability",
        text="Probability",
        color="Probability",
        color_continuous_scale="blues",
        title="Predicted Probabilities for Each Label",
        height=800,
    )
    # Add vertical line at 50% probability for threshold reference
    fig.add_hline(y=50, line_dash="dash", line_color="green")
    # Customize hover text and highlight labels above/below the threshold
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}",
        texttemplate="%{y:.2f}",
        textposition="outside",
    )

    # Display the Plotly chart
    st.plotly_chart(fig, use_container_width=True)
