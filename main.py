import minsearch
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from openai import OpenAI
import json
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import uuid
from sentence_transformers import SentenceTransformer

# environment variables
load_dotenv()
CASELAW_API_KEY = os.getenv("CASELAW_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TODO: make project configs
SEARCH_METHOD = os.getenv("SEARCH_METHOD")
MODEL_HANDLE = os.getenv("MODEL_HANDLE")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "caselaw-cases"
SIZE = 384
LIMIT = 1

HEADERS = {"Authorization": f"Token {CASELAW_API_KEY}"}

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


def display_results(results):
    print("\nTop Retrieved Cases:\n")
    for idx, res in enumerate(results):
        print(f"[{idx}] Case: {res['case_name']} | Author: {res['author']} | URL: {res['download_url']}")


def get_selected_indices(results):
    selected_indices_input = input("\nEnter indices of cases to include in context (comma-separated): ")
    selected_indices = [int(i.strip()) for i in selected_indices_input.split(",") if i.strip().isdigit()]
    return selected_indices


def build_context(results, selected_indices):
    context = ""
    for idx in selected_indices:
        if 0 <= idx < len(results):
            result = results[idx]
            context += (
                f"Case Name: {result['case_name']}\n"
                f"Opinion Type: {result['type']}\n"
                f"Case Text: {result['opinion_text']}\n\n"
            )
    return context


def get_generation_question(query):
    generation_question = input(f"\nEnter the QUESTION you'd like an answer to (press Enter to use query: '{query}'): ")
    return generation_question.strip() or query


def format_prompt(question, context):
    prompt_template = """
        You are a legal research assistant. Answer the QUESTION using only the CONTEXT provided below.
        If the CONTEXT lacks information to answer, respond accordingly and do not make assumptions.

        QUESTION: {question}

        CONTEXT:
        {context}
    """.strip()
    return prompt_template.format(question=question, context=context)


def chunk_text(text, max_tokens=500):
    """Yield chunks of text up to max_tokens words."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i+max_tokens])


def create_points_from_cases(case_data_raw):
    """Process case data and return a list of PointStruct with precomputed embeddings."""
    points = []

    for case in case_data_raw:
        opinion_text = case.get('opinion_text', '')
        
        for chunk in chunk_text(opinion_text):
            # Embed the chunk using SentenceTransformer
            embedding = EMBEDDER.encode(chunk).tolist()

            # Create PointStruct with precomputed vector
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "case_name": case.get('case_name', ''),
                    "docket_number": case.get('docket_number', ''),
                    "court_id": case.get('court_id', ''),
                    "judges": case.get('judges', ''),
                    "type": case.get('type', ''),
                    "author": case.get('author', ''),
                    "sha1": case.get('sha1', ''),
                    "download_url": case.get('download_url', '')
                }
            )
            points.append(point)
    
    return points


def vector_search(client, query, limit=1):
    query_embedding = EMBEDDER.encode(query).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True
    )
    return response.points


def fetch_opinion_text_by_docket(docket_number):
    """
    Fetches the docket by docket_number and returns the first available opinion_text.
    """
    url = "https://www.courtlistener.com/api/rest/v4/dockets/"
    resp = requests.get(url, params={"docket_number": docket_number}, headers=HEADERS)
    results = resp.json().get("results", [])
    if not results:
        return ""
    docket = results[0]
    for cluster_url in docket.get("clusters", []):
        cluster_resp = requests.get(cluster_url, headers=HEADERS)
        cluster = cluster_resp.json()
        for opinion_url in cluster.get("sub_opinions", []):
            opinion_resp = requests.get(opinion_url, headers=HEADERS)
            opinion = opinion_resp.json()
            for field in ['html_with_citations', 'html_columbia', 'html_lawbox',
                          'xml_harvard', 'html_anon_2020', 'html', 'plain_text']:
                if opinion.get(field):
                    if field.startswith('html') or field.startswith('xml'):
                        return clean_text(opinion[field])
                    else:
                        return re.sub(r'\s+', ' ', opinion[field].strip())
    return ""


def run_minsearch(docs, query):
    text_fields = ['opinion_text']
    keyword_fields = ['docket_number', 'court_id', 'judges', 'type']
    index = minsearch.Index(text_fields=text_fields, keyword_fields=keyword_fields)
    index.fit(docs)

    filters = {"court_id": "scotus", "type": "010combined"}
    boosts = {"opinion_text": 1.5}
    num_results = 5
    results = index.search(query, filter_dict=filters, boost_dict=boosts, num_results=num_results)

    if not results:
        print("No relevant documents found. Try a different query.")
        return

    display_results(results)
    selected_indices = get_selected_indices(results)
    if not selected_indices:
        print("No cases selected. Exiting.")
        return

    context = build_context(results, selected_indices)
    generation_question = get_generation_question(query)
    final_prompt = format_prompt(generation_question, context)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful legal research assistant."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2
    )
    print("\nModel Response:\n")
    print(response.choices[0].message.content.strip())


def run_vectorsearch(docs, query):
    client = QdrantClient("http://localhost:6333")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=SIZE,
                distance=models.Distance.COSINE
            )
        )
    except Exception:
        pass  # Collection may already exist

    points = create_points_from_cases(docs)
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    results = vector_search(client, query, limit=LIMIT)
    if not results:
        print("No relevant documents found. Try a different query.")
        return

    top_result = results[0].payload
    docket_number = top_result.get('docket_number')
    print(f"Top result: {top_result.get('case_name')} ({docket_number})")
    print(f"URL: {top_result.get('download_url')}")

    opinion_text = fetch_opinion_text_by_docket(docket_number)
    if not opinion_text:
        print("Could not fetch opinion text for the selected docket.")
        return

    context = f"Case Name: {top_result.get('case_name')}\nOpinion Type: {top_result.get('type')}\nCase Text: {opinion_text}\n"
    generation_question = get_generation_question(query)
    final_prompt = format_prompt(generation_question, context)

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful legal research assistant."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2
    )
    print("\nModel Response:\n")
    print(response.choices[0].message.content.strip())


def main():
    all_data = process_docket(court='scotus', num_dockets=3)
    docs = flatten_opinions(all_data)

    with open("cases.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    query = input("Please enter your search query: ")

    if SEARCH_METHOD == "minsearch":
        run_minsearch(docs, query)
    elif SEARCH_METHOD == "vectorsearch":
        run_vectorsearch(docs, query)
    else:
        print("Invalid SEARCH_METHOD specified.")

if __name__ == "__main__":
    main()
