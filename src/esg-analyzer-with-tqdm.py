import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
from tqdm import tqdm, trange

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NLI model and tokenizer
MODEL_NAME = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

# Load sentence transformer for embeddings
#embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
#embedder = SentenceTransformer('msmarco-distilbert-base-v4').to(device)
#embedder = SentenceTransformer('all-mpnet-base-v2').to(device)
#embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

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

print(f"Pre-embedding {len(topics)} ESG topics...")
# Pre-embed all topics once at startup
topic_embeddings = embedder.encode(
    topics,
    convert_to_tensor=True,
    device=device,
    batch_size=16,
    show_progress_bar=True
)

# Entailment threshold for NLI
threshold = 0.5
BATCH_SIZE = 8  # Number of NLI inferences to process in one batch

def process_report(report_text, similarity_threshold=0.5, fixed_topk=5, chunk_size=512, overlap=256):
    # Tokenize the report into words using NLTK
    words = nltk.tokenize.word_tokenize(report_text)

    # Create overlapping chunks of words
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        if len(chunk) == 0:
            break
        chunks.append(" ".join(chunk))

    print(f"Number of chunks: {len(chunks)}")

    # Generate embeddings for all chunks with progress bar
    print("Encoding document chunks...")
    chunk_embeddings = embedder.encode(
        chunks, 
        convert_to_tensor=True, 
        device=device,
        show_progress_bar=True
    )

    # Prepare data for batched NLI inference
    all_premises = []
    all_hypotheses = []
    all_similarities = []  # Store cosine similarity scores for each chunk
    chunk_indices_per_topic = []

    # For each topic, select relevant chunks using top-K strategy with progress bar
    print("Finding relevant chunks for each ESG topic...")
    for idx in tqdm(range(len(topics)), desc="Processing ESG topics"):
        topic = topics[idx]
        topic_embedding = topic_embeddings[idx]
        similarities = util.pytorch_cos_sim(chunk_embeddings, topic_embedding).squeeze()

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

    # Batch NLI inference with progress bar
    print("Running NLI inference...")
    answers = ["0"] * len(topics)
    confidence_scores = [0.0] * len(topics)  # Will store max aggregated scores
    entailment_probs_all = []

    for i in tqdm(range(0, len(all_premises), BATCH_SIZE), desc="NLI batch processing"):
        batch_premises = all_premises[i:i + BATCH_SIZE]
        batch_hypotheses = all_hypotheses[i:i + BATCH_SIZE]

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
        # For BART models:
        entailment_probs = probs[:, 2].cpu().tolist()
        entailment_probs_all.extend(entailment_probs)

    # Combine entailment probabilities with cosine similarity scores using weighted aggregation:
    # aggregated_score = 0.7 * entail_prob + 0.3 * cosine_similarity
    aggregated_scores_all = [0.7 * e + 0.3 * s for e, s in zip(entailment_probs_all, all_similarities)]

    # Assign aggregated scores back to topics
    print("Calculating final scores...")
    prob_idx = 0
    for topic_idx, num_chunks in chunk_indices_per_topic:
        if num_chunks > 0:
            topic_aggregated_scores = aggregated_scores_all[prob_idx:prob_idx + num_chunks]
            max_aggregated_score = max(topic_aggregated_scores)
            answers[topic_idx] = "1" if max_aggregated_score > threshold else "0"
            confidence_scores[topic_idx] = max_aggregated_score
            prob_idx += num_chunks

    return answers, confidence_scores

def main():
    REPORTS_DIR = "annual_reports/"
    
    # Get list of files to process
    file_list = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".txt")]
    
    results = []
    
    # Process reports with progress bar for files
    for filename in tqdm(file_list, desc="Processing annual reports"):
        parts = filename.split("_")
        company = parts[0]
        year = parts[1].split(".")[0]
        
        print(f"\nProcessing {company} ({year})...")
        report_path = os.path.join(REPORTS_DIR, filename)
        
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
            
        answers, confidence_scores = process_report(
            report_text,
            similarity_threshold=0.3,
            fixed_topk=5
        )
        
        esg_score = sum(1 for ans in answers if ans == "1")
        print(f"ESG Score for {company} ({year}): {esg_score}/{len(topics)}")
        
        # Include both answers and confidence scores in results
        results.append([company, year] + answers + confidence_scores + [esg_score])

    # Save results to CSV with confidence scores
    print("Saving results to CSV...")
    columns = (
        ["company", "year"] +
        [f"q{i+1}" for i in range(len(topics))] +  # Binary answers
        [f"conf_q{i+1}" for i in range(len(topics))] +  # Confidence (aggregated) scores
        ["esg_score"]
    )
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("esg_scores_with_confidence.csv", index=False)

    print("Processing complete. Results saved to 'esg_scores_with_confidence.csv'.")

if __name__ == "__main__":
    main()
