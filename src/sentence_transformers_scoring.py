"""
This module processes annual reports to determine the disclosure of various ESG (Environmental, Social, and Governance) topics.

It utilizes Natural Language Inference (NLI) and sentence embeddings to analyze the text of the reports. The module performs the following steps:
1. Downloads necessary NLTK data for sentence tokenization.
2. Sets up the device for computation (GPU if available).
3. Loads pre-trained models for NLI and sentence embeddings.
4. Defines a list of ESG topics to be checked in the reports.
5. Processes each report by:
   - Tokenizing the report into sentences.
   - Creating overlapping chunks of sentences.
   - Generating embeddings for the chunks.
   - Computing cosine similarity between topic embeddings and chunk embeddings.
   - Applying NLI to the most relevant chunks to determine if the topic is disclosed.
6. Aggregates the results and saves them to a CSV file.

The results include binary answers indicating the disclosure of each topic and an overall ESG score for each report.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk

# Download necessary NLTK data for sentence tokenization
nltk.download("punkt")

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NLI model and tokenizer
MODEL_NAME = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

# Load sentence transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
#embedder = SentenceTransformer('msmarco-distilbert-base-v4').to(device)
#embedder = SentenceTransformer('all-mpnet-base-v2').to(device)
#embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
# embedder = SentenceTransformer(
#     "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
# ).to(device)

# List of ESG topics
topics = [
    "Nitrogen Oxide Emissions",
    "VOC Emissions",
    "Carbon Monoxide Emissions",
    "Particulate Emissions",
    "Sulphur Dioxide / Sulphur Oxide Emissions",
    "Emissions Reduction Initiatives",
    "Climate Change Policy",
    "Climate Change Opportunities Discussed",
    "Risks of Climate Change Discussed",
    "Direct CO2 Emissions",
    "Indirect CO2 Emissions",
    "ODS Emissions",
    "GHG Scope 1",
    "GHG Scope 2",
    "GHG Scope 3",
    "Scope 2 Market Based GHG Emissions",
    "Scope of Disclosure",
    "Carbon per Unit of Production",
    "Biodiversity Policy",
    "Number of Environmental Fines",
    "Environmental Fines (Amount)",
    "Number of Significant Environmental Fines",
    "Amount of Significant Environmental Fines",
    "Energy Efficiency Policy",
    "Total Energy Consumption",
    "Renewable Energy Use",
    "Electricity Used",
    "Fuel Used - Coal/Lignite",
    "Fuel Used - Natural Gas",
    "Fuel Used - Crude Oil/Diesel",
    "Self Generated Renewable Electricity",
    "Energy Per Unit of Production",
    "Waste Reduction Policy",
    "Hazardous Waste",
    "Total Waste",
    "Waste Recycled",
    "Raw Materials Used",
    "% Recycled Materials",
    "Waste Sent to Landfills",
    "Percentage Raw Material from Sustainable Sources",
    "Environmental Supply Chain Management",
    "Water Policy",
    "Total Water Discharged",
    "Water per Unit of Production",
    "Total Water Withdrawal",
    "Water Consumption",
    "Human Rights Policy",
    "Policy Against Child Labor",
    "Quality Assurance and Recall Policy",
    "Consumer Data Protection Policy",
    "Community Spending",
    "Number of Customer Complaints",
    "Total Corporate Foundation and Other Giving",
    "Equal Opportunity Policy",
    "Gender Pay Gap Breakout",
    "% Women in Management",
    "% Women in Workforce",
    "% Minorities in Management",
    "% Minorities in Workforce",
    "% Disabled in Workforce",
    "Percentage Gender Pay Gap for Senior Management",
    "Percentage Gender Pay Gap Mid & Other Management",
    "Percentage Gender Pay Gap Employees Ex Management",
    "% Gender Pay Gap Tot Empl Including Management",
    "% Women in Middle and or Other Management",
    "Business Ethics Policy",
    "Anti-Bribery Ethics Policy",
    "Political Donations",
    "Health and Safety Policy",
    "Fatalities - Contractors",
    "Fatalities - Employees",
    "Fatalities - Total",
    "Lost Time Incident Rate",
    "Total Recordable Incident Rate",
    "Lost Time Incident Rate - Contractors",
    "Total Recordable Incident Rate - Contractors",
    "Total Recordable Incident Rate - Workforce",
    "Lost Time Incident Rate - Workforce",
    "Training Policy",
    "Fair Renumeration Policy",
    "Number of Employees - CSR",
    "Employee Turnover %",
    "% Employees Unionized",
    "Employee Training Cost",
    "Total Hours Spent by Firm - Employee Training",
    "Number of Contractors",
    "Social Supply Chain Management",
    "Number of Suppliers Audited",
    "Number of Supplier Audits Conducted",
    "Number Supplier Facilities Audited",
    "Percentage of Suppliers in Non-Compliance",
    "Percentage Suppliers Audited",
    "Audit Committee Meetings",
    "Years Auditor Employed",
    "Size of Audit Committee",
    "Number of Independent Directors on Audit Committee",
    "Audit Committee Meeting Attendance Percentage",
    "Company Conducts Board Evaluations",
    "Size of the Board",
    "Number of Board Meetings for the Year",
    "Board Meeting Attendance %",
    "Number of Executives / Company Managers",
    "Number of Non Executive Directors on Board",
    "Company Has Executive Share Ownership Guidelines",
    "Director Share Ownership Guidelines",
    "Size of Compensation Committee",
    "Num of Independent Directors on Compensation Cmte",
    "Number of Compensation Committee Meetings",
    "Compensation Committee Meeting Attendance %",
    "Number of Independent Directors",
    "Size of Nomination Committee",
    "Num of Independent Directors on Nomination Cmte",
    "Number of Nomination Committee Meetings",
    "Nomination Committee Meeting Attendance Percentage",
    "Verification Type",
    "Employee CSR Training",
    "Board Duration (Years)",
]

# Entailment threshold for NLI
threshold = 0.5


# Function to process a single report
def process_report(report_text):
    """
    Process a single annual report to determine if it discloses information about each ESG topic.

    Args:
        report_text (str): The full text of the annual report.

    Returns:
        list: Binary answers ("1" or "0") indicating whether each topic is disclosed.
    """
    # Tokenize the report into sentences using NLTK
    sentences = nltk.tokenize.sent_tokenize(report_text)

    # Create overlapping chunks of sentences
    chunk_size = 5  # Number of sentences per chunk
    overlap = 2  # Number of overlapping sentences between consecutive chunks
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + chunk_size]
        if len(chunk) == 0:
            break
        chunks.append(" ".join(chunk))

    # Log the number of chunks for monitoring
    print(f"Number of chunks: {len(chunks)}")

    # Generate embeddings for all chunks
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)

    answers = []
    for topic in topics:
        # Embed the topic
        topic_embedding = embedder.encode(topic, convert_to_tensor=True, device=device)

        # Compute cosine similarity between topic and chunks
        similarities = util.pytorch_cos_sim(chunk_embeddings, topic_embedding).squeeze()

        # Get top-5 most similar chunks
        k = min(5, len(chunks))
        top_k_indices = similarities.topk(k=k).indices
        rel_chunks = [chunks[i] for i in top_k_indices]

        if not rel_chunks:
            answers.append("0")
            continue

        # Apply NLI to top-k chunks
        hq = f"This text discloses information about {topic}."
        premises = rel_chunks
        hypotheses = [hq] * len(rel_chunks)
        inputs = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        entailment_probs = probs[:, 2].cpu().tolist()  # Entailment class
        max_entailment = max(entailment_probs) if entailment_probs else 0.0
        answer = "1" if max_entailment > threshold else "0"
        answers.append(answer)
    return answers


# Process all reports in the directory
REPORTS_DIR = "../parsed/annual_reports/hdfc"
results = []
for filename in os.listdir(REPORTS_DIR):
    if filename.endswith(".txt"):
        parts = filename.split("_")
        company = parts[0]
        year = parts[1].split(".")[0]
        with open(os.path.join(REPORTS_DIR, filename), "r", encoding="utf-8") as f:
            report_text = f.read()
            print(os.path.join(REPORTS_DIR, filename))
        answers = process_report(report_text)
        esg_score = sum(1 for ans in answers if ans == "1")
        results.append([company, year] + answers + [esg_score])

# Save results to CSV
columns = ["company", "year"] + [f"q{i+1}" for i in range(len(topics))] + ["esg_score"]
df = pd.DataFrame(results, columns=columns)
df.to_csv("esg_scores.csv", index=False)

print("Processing complete. Results saved to 'esg_scores.csv'.")
