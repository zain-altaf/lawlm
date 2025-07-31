import requests
import os
import json
import re

from dotenv import load_dotenv
from bs4 import BeautifulSoup

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
CASELAW_API_KEY = os.getenv("CASELAW_API_KEY")

# --- API Headers ---
HEADERS = {"Authorization": f"Token {CASELAW_API_KEY}"}

# --- Data Fetching and Processing ---

def clean_text(content: str) -> str:
    """Strips HTML/XML tags and normalizes whitespace."""
    if not content:
        return ''
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)

def process_docket(court: str = 'scotus', num_dockets: int = 5) -> list:
    """
    Fetches dockets, clusters, and opinions from CourtListener API,
    returning a simplified list of dictionaries.
    """
    print(f"Fetching {num_dockets} dockets from court: {court}...")
    docket_resp = requests.get(
        "https://www.courtlistener.com/api/rest/v4/dockets/",
        params={"court": court, "page_size": num_dockets},
        headers=HEADERS
    )
    docket_resp.raise_for_status()
    dockets = docket_resp.json().get("results", [])

    all_data = []
    for docket in dockets:
        for cluster_url in docket.get("clusters", []):
            cluster_resp = requests.get(cluster_url, headers=HEADERS)
            cluster = cluster_resp.json()
            for opinion_url in cluster.get("sub_opinions", []):
                opinion_resp = requests.get(opinion_url, headers=HEADERS)
                opinion = opinion_resp.json()
                text, source = "", 'Unknown'
                for field in ['html_with_citations', 'html_columbia', 'html_lawbox', 'xml_harvard', 'html', 'plain_text']:
                    if opinion.get(field):
                        text = clean_text(opinion[field]) if 'html' in field or 'xml' in field else re.sub(r'\s+', ' ', opinion[field].strip())
                        source = field
                        break
                all_data.append({
                    "id": opinion.get("id"),
                    "docket_number": docket.get("docket_number"),
                    "case_name": cluster.get("case_name"),
                    "court_id": docket.get("court_id"),
                    "judges": cluster.get("judges"),
                    "author": opinion.get("author_str"),
                    "type": opinion.get("type"),
                    "sha1": opinion.get("sha1"),
                    "download_url": opinion.get("download_url"),
                    "opinion_text": text,
                    "source_field": source
                })
    print(f"Successfully processed {len(all_data)} opinions.")
    return all_data

def main():
    """Main function to run the data ingestion process."""
    print("Starting raw data ingestion process...")
    
    # Get command line arguments for court and number of dockets
    import argparse
    parser = argparse.ArgumentParser(description='Fetch raw legal case data')
    parser.add_argument('--court', default='scotus', help='Court identifier (default: scotus)')
    parser.add_argument('--num_dockets', type=int, default=5, help='Number of dockets to fetch (default: 5)')
    parser.add_argument('--output', default='data/raw_cases.json', help='Output filename (default: data/raw_cases.json)')
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Fetch and process documents
    docs = process_docket(court=args.court, num_dockets=args.num_dockets)
    
    # Save raw documents to JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} raw documents to {args.output}")

    print("\nRaw data ingestion completed successfully!")
    print(f"- Raw documents saved to: {args.output}")
    print("- Next step: Run 'python process_pipeline.py' to prepare data for search")

if __name__ == "__main__":
    main()