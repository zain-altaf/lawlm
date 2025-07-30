import minsearch
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from openai import OpenAI

# environment variables
load_dotenv()
CASELAW_API = os.getenv("CASELAW_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Token {CASELAW_API}"}

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def clean_text(content):
    """
    Strips HTML/XML tags and normalizes whitespace.
    """
    if content is None:
        return ''
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text


def process_docket(court='scotus', num_dockets=5):
    """
    Fetches dockets, clusters, and opinions from CourtListener.
    Simplifies JSON to essential fields and cleans opinion texts.
    """
    docket_resp = requests.get(
        "https://www.courtlistener.com/api/rest/v4/dockets/",
        params={"court": court, "page_size": num_dockets},
        headers=HEADERS
    )
    dockets = docket_resp.json()["results"]

    all_data = []

    for docket in dockets:
        docket_data = {
            "id": docket.get("id"),
            "case_name": docket.get("case_name"),
            "docket_number": docket.get("docket_number"),
            "absolute_url": docket.get("absolute_url"),
            "court_id": docket.get("court_id"),
            "date_filed": docket.get("date_filed"),
            "clusters": []
        }

        # Fetch clusters
        for cluster_url in docket.get("clusters", []):
            cluster_resp = requests.get(cluster_url, headers=HEADERS)
            cluster = cluster_resp.json()

            cluster_data = {
                "id": cluster.get("id"),
                "case_name": cluster.get("case_name"),
                "judges": cluster.get("judges"),
                "date_filed": cluster.get("date_filed"),
                "precedential_status": cluster.get("precedential_status"),
                "opinions": []
            }

            # Fetch opinions
            for opinion_url in cluster.get("sub_opinions", []):
                opinion_resp = requests.get(opinion_url, headers=HEADERS)
                opinion = opinion_resp.json()

                # Prioritize opinion text source
                opinions_text = ''
                source = 'Unknown'
                for field in ['html_with_citations', 'html_columbia', 'html_lawbox',
                              'xml_harvard', 'html_anon_2020', 'html', 'plain_text']:
                    if opinion.get(field):
                        opinions_text = opinion[field]
                        source = field
                        break

                # Clean if HTML/XML, else strip whitespace for plain_text
                if source.startswith('html') or source.startswith('xml'):
                    opinions_text = clean_text(opinions_text)
                else:
                    opinions_text = re.sub(r'\s+', ' ', opinions_text.strip())

                opinion_data = {
                    "id": opinion.get("id"),
                    "author_str": opinion.get("author_str"),
                    "type": opinion.get("type"),
                    "sha1": opinion.get("sha1"),
                    "download_url": opinion.get("download_url"),
                    "text_source": source,
                    "opinion_text": opinions_text  # You can truncate here if needed
                }

                cluster_data["opinions"].append(opinion_data)

            docket_data["clusters"].append(cluster_data)

        all_data.append(docket_data)

    return all_data


def flatten_opinions(all_data):
    """
    Flattens docket-cluster-opinion JSON into a list of documents for indexing.
    """
    docs = []
    for docket in all_data:
        docket_number = docket.get("docket_number")
        court_id = docket.get("court_id")
        case_name = docket.get("case_name")
        for cluster in docket.get("clusters", []):
            judges = cluster.get("judges")
            for opinion in cluster.get("opinions", []):
                doc = {
                    "docket_number": docket_number,
                    "court_id": court_id,
                    "case_name": case_name,
                    "judges": judges,
                    "type": opinion.get("type"),
                    "author": opinion.get("author_str"),
                    "sha1": opinion.get("sha1"),
                    "download_url": opinion.get("download_url"),
                    "opinion_text": opinion.get("opinion_text")
                }
                docs.append(doc)
    return docs


if __name__ == "__main__":
    all_data = process_docket(court='scotus', num_dockets=5)
    docs = flatten_opinions(all_data)

    text_fields = ['opinion_text']
    keyword_fields = ['docket_number', 'court_id', 'judges', 'type']
    index = minsearch.Index(text_fields=text_fields, keyword_fields=keyword_fields)
    index.fit(docs)

    query = input("Please enter your search query: ")

    filters = {"court_id": "scotus", "type": "010combined"}
    boosts = {"opinion_text": 1.5}
    num_results = 5
    results = index.search(query, filter_dict=filters, boost_dict=boosts, num_results=num_results)

    if not results:
        print("No relevant documents found. Try a different query.")
        exit()

    print("\nTop Retrieved Cases:\n")
    for idx, res in enumerate(results):
        print(f"[{idx}] Case: {res['case_name']} | Author: {res['author']} | URL: {res['download_url']}")

    selected_indices_input = input("\nEnter indices of cases to include in context (comma-separated): ")
    selected_indices = [int(i.strip()) for i in selected_indices_input.split(",") if i.strip().isdigit()]

    if not selected_indices:
        print("No cases selected. Exiting.")
        exit()

    context = ""
    for idx in selected_indices:
        if 0 <= idx < len(results):
            result = results[idx]
            context += (
                f"Case Name: {result['case_name']}\n"
                f"Opinion Type: {result['type']}\n"
                f"Case Text: {result['opinion_text']}\n\n"
            )

    generation_question = input(f"\nEnter the QUESTION you'd like an answer to (press Enter to use query: '{query}'): ")
    if not generation_question.strip():
        generation_question = query

    prompt_template = """
        You are a legal research assistant. Answer the QUESTION using only the CONTEXT provided below.
        If the CONTEXT lacks information to answer, respond accordingly and do not make assumptions.

        QUESTION: {question}

        CONTEXT:
        {context}
    """.strip()

    final_prompt = prompt_template.format(
        question=generation_question,
        context=context
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",  # Use "gpt-3.5-turbo" for cost efficiency if needed
        messages=[
            {"role": "system", "content": "You are a helpful legal research assistant."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2
    )

    print("\nModel Response:\n")
    print(response.choices[0].message.content.strip())
