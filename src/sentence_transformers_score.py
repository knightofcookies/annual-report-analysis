""" This script uses Sentence Transformers to score the ESG disclosure of annual reports. """

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk

# Download necessary NLTK data for sentence tokenization
nltk.download("punkt")
nltk.download("punkt_tab")

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NLI model and tokenizer
MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

# Load sentence transformer for embeddings
# embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
# embedder = SentenceTransformer('msmarco-distilbert-base-v4').to(device)
# embedder = SentenceTransformer('all-mpnet-base-v2').to(device)
# embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
# embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
embedder = SentenceTransformer("ProsusAI/finbert").to(device)

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

# Pre-embed all topics once at startup
topic_embeddings = embedder.encode(
    topics, convert_to_tensor=True, device=device, batch_size=32
)

# Entailment threshold for NLI
THRESHOLD = 0.5
BATCH_SIZE = 32  # Number of NLI inferences to process in one batch


def process_report(
    report_text, dynamic_selection=True, similarity_threshold=0.5, fixed_topk=5
):
    """
    Process a single annual report to determine if it discloses information about each ESG topic.
    Uses batched NLI inference and returns both binary answers and confidence scores.

    Two selection strategies are available for choosing relevant text chunks:
      1. Fixed top-K: always select the top-k (default 5) chunks per topic.
      2. Dynamic selection: select all chunks with a cosine similarity score above the threshold.

    Args:
        report_text (str): The full text of the annual report.
        dynamic_selection (bool): If True, use the similarity threshold method; if False, use fixed top-K.
        similarity_threshold (float): The cosine similarity threshold (used only if dynamic_selection=True).
        fixed_topk (int): Number of top similar chunks to select (used only if dynamic_selection=False).

    Returns:
        tuple: (list of binary answers ("1" or "0"), list of confidence scores)
    """
    # Tokenize the report into sentences using NLTK
    sentences = nltk.tokenize.sent_tokenize(report_text)

    # Create overlapping chunks of sentences
    chunk_size = 5
    overlap = 2
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + chunk_size]
        if len(chunk) == 0:
            break
        chunks.append(" ".join(chunk))

    print(f"Number of chunks: {len(chunks)}")

    # Generate embeddings for all chunks
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)

    # Prepare data for batched NLI inference
    all_premises = []
    all_hypotheses = []
    all_similarities = []  # Store cosine similarity scores for each chunk
    chunk_indices_per_topic = []

    # For each topic, select relevant chunks using one of the two strategies
    for idx, topic in enumerate(topics):
        topic_embedding = topic_embeddings[idx]
        similarities = util.pytorch_cos_sim(chunk_embeddings, topic_embedding).squeeze()

        if dynamic_selection:
            # Dynamic selection: only use chunks above a certain similarity threshold.
            valid_indices = (similarities > similarity_threshold).nonzero(
                as_tuple=True
            )[0]
            if len(valid_indices) > 0:
                # Sort valid indices by similarity in descending order
                valid_similarities = similarities[valid_indices]
                sorted_order = torch.argsort(valid_similarities, descending=True)
                top_indices = valid_indices[sorted_order]
                top_similarities = valid_similarities[sorted_order].tolist()
            else:
                top_indices = []
                top_similarities = []
        else:
            # Fixed top-K selection: always select the top 'fixed_topk' chunks.
            k = min(fixed_topk, len(chunks))
            topk = similarities.topk(k=k)
            top_indices = topk.indices
            top_similarities = topk.values.tolist()

        rel_chunks = [chunks[i] for i in top_indices]
        hypothesis = f"This text discloses information about {topic}."
        all_premises.extend(rel_chunks)
        all_hypotheses.extend([hypothesis] * len(rel_chunks))
        all_similarities.extend(top_similarities)
        chunk_indices_per_topic.append((idx, len(rel_chunks)))

    if not all_premises:
        return (["0"] * len(topics), [0.0] * len(topics))

    # Batch NLI inference
    answers = ["0"] * len(topics)
    confidence_scores = [0.0] * len(topics)  # Will store max aggregated scores
    entailment_probs_all = []

    for i in range(0, len(all_premises), BATCH_SIZE):
        batch_premises = all_premises[i : i + BATCH_SIZE]
        batch_hypotheses = all_hypotheses[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        entailment_probs = probs[:, 0].cpu().tolist()  # Entailment class for RoBERTa
        # For BART models you might use: entailment_probs = probs[:, 2].cpu().tolist()
        entailment_probs_all.extend(entailment_probs)

    # Combine entailment probabilities with cosine similarity scores using weighted aggregation:
    # aggregated_score = 0.7 * entail_prob + 0.3 * cosine_similarity
    aggregated_scores_all = [
        0.7 * e + 0.3 * s for e, s in zip(entailment_probs_all, all_similarities)
    ]

    # Assign aggregated scores back to topics
    prob_idx = 0
    for topic_idx, num_chunks in chunk_indices_per_topic:
        if num_chunks > 0:
            topic_aggregated_scores = aggregated_scores_all[
                prob_idx : prob_idx + num_chunks
            ]
            max_aggregated_score = max(topic_aggregated_scores)
            answers[topic_idx] = "1" if max_aggregated_score > THRESHOLD else "0"
            confidence_scores[topic_idx] = max_aggregated_score
            prob_idx += num_chunks

    return answers, confidence_scores


USE_DYNAMIC_SELECTION = False
REPORTS_DIR = "hdfc/"


def main():
    """Main function to process all annual reports in the REPORTS_DIR directory."""
    # Toggle between dynamic selection and fixed top-K selection:
    # Set USE_DYNAMIC_SELECTION = True to use the threshold-based dynamic selection,
    # or False to use fixed top-K selection.
    # <-- change this flag as needed

    results = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".txt"):
            parts = filename.split("_")
            company = parts[0]
            year = parts[1].split(".")[0]
            with open(os.path.join(REPORTS_DIR, filename), "r", encoding="utf-8") as f:
                report_text = f.read()
                print(os.path.join(REPORTS_DIR, filename))
            answers, confidence_scores = process_report(
                report_text,
                dynamic_selection=USE_DYNAMIC_SELECTION,
                similarity_threshold=0.3,  # Adjust this value if needed
                fixed_topk=5,  # Adjust this value if needed
            )
            esg_score = sum(1 for ans in answers if ans == "1")
            # Include both answers and confidence scores in results
            results.append([company, year] + answers + confidence_scores + [esg_score])

    # Save results to CSV with confidence scores
    columns = (
        ["company", "year"]
        + [f"q{i+1}" for i in range(len(topics))]  # Binary answers
        + [f"conf_q{i+1}" for i in range(len(topics))]  # Confidence (aggregated) scores
        + ["esg_score"]
    )
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("esg_scores_with_confidence.csv", index=False)

    print("Processing complete. Results saved to 'esg_scores_with_confidence.csv'.")


if __name__ == "__main__":
    main()
