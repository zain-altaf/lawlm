### fetch_court_data.py is integral for ingesting court data

from dotenv import load_dotenv
import os
import logging
import urllib
import requests
import json

from opinion_utills import enhanced_text_processing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv('CASELAW_API_KEY')
headers = {'Authorization': f'Token {api_key}'} if api_key else {}

def get_dockets(num_pages=2, court='scotus'):
    '''Fetch court data from CourtListener and store it locally.'''
    
    new_dockets = []
    page_count = 0
    # consecutive_empty_pages = 0
    # max_consecutive_empty = 50
    
    base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"

    cursor = None

    while page_count < num_pages:
        page_count += 1
                
        try:
            params = {
                "court": court,
                "ordering": "id"  # Ordering by id for consistent pagination
            }
            if cursor:
                params["cursor"] = cursor
            
            # Construct full URL with params for retry function
            query_string = urllib.parse.urlencode(params)
            full_url = f"{base_url}?{query_string}"
        
            # Fetch data and throw error if gotten
            try:
                response = requests.get(full_url, headers=headers)
                response_data = response.json()
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt. Please try again later. Error: {e}")

            # results contains all 20 dockets for one page
            page_dockets = response_data.get('results', [])

            # get all 20 dockets for this page
            for docket in page_dockets:
                docket_id = docket.get('id', '')
                if not docket_id:
                    raise ValueError(f"Docket without ID found on page {page_count} (cursor={cursor})")
                
                # add page cursor for later debugging in case there are issues 
                docket["page_cursor"] = cursor
                new_dockets.append(docket)

            logger.info(f"📊 Page {page_count}, added {len(page_dockets)} dockets. {len(new_dockets)} new dockets added this session.")
            
            # prepare for the next page and ensure it's there
            next_url = response_data.get('next')
            if next_url:
                parsed = urllib.parse.urlparse(next_url)
                query_params = urllib.parse.parse_qs(parsed.query)
                cursor = query_params.get('cursor', [None])[0]
            else:
                cursor = None

            # Check if we've reached the end of available data
            if not cursor:
                logger.info(f"📋 Reached end of available dockets (no more pages)")
                break
            
        except Exception as e:
            logger.error(f"❌ Unexpected error on page {page_count}: {e}")
            break
    
    logger.info(f"Docket Ingestion for Session Complete:")
    logger.info(f"Pages Fetched: {page_count}")
    logger.info(f"Total Number of Dockets Found: {len(new_dockets)}")
    
    return new_dockets


def get_clusters_and_opinions(docket):
    '''From dockets, get all clusters and opinions.'''
    
    opinions = []
    clusters = []
    docket_id = docket.get('id', '')

    if not docket_id:
        raise ValueError(f"Docket without ID found with cursor: {docket['page_cursor']}")

    logger.info(f"Fetching clusters and opinions for docket {docket_id}")

    for cluster_url in docket.get('clusters', []):
        try:
            logger.debug(f"Processing cluster: {cluster_url}")

            cluster = requests.get(cluster_url, headers=headers)
            cluster_data = cluster.json()

            if cluster_data is None:
                logger.error(f"Failed to fetch cluster {cluster_url}")
                continue

            clusters.append(cluster_data)
            cluster_id = cluster_data.get('id', '')

            for opinion_url in cluster_data.get("sub_opinions", []):
                try:
                    logger.debug(f"Processing opinion: {opinion_url}")

                    opinion = requests.get(opinion_url, headers=headers)
                    opinion_data = opinion.json()

                    if opinion_data is None:
                        logger.warning(f"Failed to fetch opinion {opinion_url}")
                        continue
                    
                    # Extract text from available fields in order of recommendation
                    raw_text = None
                    source_field = None
                    for field in [
                        'html_with_citations',
                        'plain_text',
                        'html_columbia',
                        'html_lawbox',
                        'html_anon_2020',
                        'html'
                    ]:
                        if opinion_data.get(field):
                            raw_text = opinion_data[field]
                            source_field = field
                            break

                    if not raw_text or len(raw_text.strip()) < 100:
                        logger.debug(f"Skipping opinion {opinion_data.get('id')} - insufficient text")
                        continue

                    # Process text using existing enhanced processing
                    try:
                        processed = enhanced_text_processing(raw_text)

                        opinion_record = {
                            "docket_id": docket_id,
                            "cluster_id": cluster_id,
                            "opinion_id": opinion_data.get("id"),
                            "case_name": cluster_data.get("case_name", "Unknown Case"),
                            "court_id": docket.get("court_id", "unknown"),
                            "judges": cluster_data.get("judges", ""),
                            "author": opinion_data.get("author_id", ""),
                            "opinion_type": opinion_data.get("type", "unknown"),
                            "date_filed": cluster_data.get("date_filed"),
                            "precedential_status": cluster_data.get("precedential_status"),
                            "sha1": opinion_data.get("sha1"),
                            "download_url": opinion_data.get("download_url"),
                            "source_field": source_field,
                            "opinion_text": processed['cleaned_text'],
                            "citations": processed['citations'],
                            "legal_entities": processed['legal_entities'],
                            "text_stats": processed['text_stats'],
                            "date_created": opinion_data.get("date_created"),
                            "date_modified": opinion_data.get("date_modified"),
                        }
                        
                        opinions.append(opinion_record)
                        logger.debug(f"Successfully processed opinion {opinion_data.get('id')}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process text for opinion {opinion_data.get('id')}: {e}")
                        continue
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch opinion {opinion_url}: {e}")
                    continue
                
        except Exception as e:
            logger.warning(f"Failed to fetch cluster {cluster_url}: {e}")
            continue

    logger.info(f"📄 Found {len(opinions)} opinions and {len(clusters)} clusters in docket {docket_id}")
    return clusters, opinions


def main(num_pages=1):
    # fetch dockets from scotus
    dockets = get_dockets(num_pages=num_pages, court='scotus')
    print("dockets:", len(dockets))

    all_opinions = []

    # get all clusters and opinions from dockets
    for _, docket in enumerate(dockets, 1):

        docket_id = docket.get('id', '')
        if not docket_id:
            raise ValueError(f"Docket without ID found with cursor: {docket['page_cursor']})")

        _, docket_opinions = get_clusters_and_opinions(docket)
        all_opinions.extend(docket_opinions)

    # save to local json file
    output_file = os.getenv('OUTPUT_FILE', '/app/output/opinions.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_opinions, f)
    logger.info(f"✅ Saved {len(all_opinions)} opinions to {output_file}")

if '__main__' == __name__:
    main(num_pages=1)