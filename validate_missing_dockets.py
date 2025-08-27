#!/usr/bin/env python3
"""
Validate which dockets didn't produce vectors and why.
"""
import os
import requests
from typing import Set, Dict, Any, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def get_existing_dockets() -> Set[str]:
    """Get existing docket numbers from Qdrant."""
    client = QdrantClient(url="http://localhost:6333")
    existing_dockets = set()
    offset = None
    
    while True:
        points, next_offset = client.scroll(
            collection_name="caselaw-chunks-hybrid",
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            break
            
        for point in points:
            docket_number = point.payload.get('docket_number')
            if docket_number:
                existing_dockets.add(docket_number)
                
        offset = next_offset
        if next_offset is None:
            break
    
    return existing_dockets

def fetch_and_analyze_recent_dockets(court: str = "scotus", num_dockets: int = 25) -> Dict[str, Any]:
    """Fetch recent dockets and analyze their content availability."""
    CASELAW_API_KEY = os.getenv('CASELAW_API_KEY')
    HEADERS = {'Authorization': f'Token {CASELAW_API_KEY}'} if CASELAW_API_KEY else {}
    
    print(f"🔍 Fetching {num_dockets} recent dockets from {court}...")
    
    # Get existing dockets first
    existing_dockets = get_existing_dockets()
    print(f"📊 Found {len(existing_dockets)} existing dockets in Qdrant")
    
    # Fetch recent dockets from API
    all_dockets = []
    page = 1
    
    while len(all_dockets) < num_dockets + 10:  # Get a few extra to ensure we have enough
        response = requests.get(
            "https://www.courtlistener.com/api/rest/v4/dockets/",
            params={
                "court": court, 
                "page_size": 200,
                "page": page
            },
            headers=HEADERS,
            timeout=30
        )
        response.raise_for_status()
        
        page_dockets = response.json().get('results', [])
        if not page_dockets:
            break
            
        all_dockets.extend(page_dockets)
        page += 1
        
        if len(page_dockets) < 200:  # End of results
            break
    
    print(f"📄 Fetched {len(all_dockets)} total dockets from API")
    
    # Analyze each docket
    analysis_results = {
        'in_qdrant': [],
        'not_in_qdrant': [],
        'analysis': []
    }
    
    dockets_to_analyze = all_dockets[:num_dockets * 2]  # Analyze more than needed
    
    for idx, docket in enumerate(dockets_to_analyze):
        docket_number = docket.get('docket_number', '')
        if not docket_number:
            continue
            
        print(f"[{idx+1}/{len(dockets_to_analyze)}] Analyzing docket: {docket_number}")
        
        # Check if in Qdrant
        in_qdrant = docket_number in existing_dockets
        
        # Analyze docket content
        clusters = docket.get("clusters", [])
        docket_analysis = {
            'docket_number': docket_number,
            'in_qdrant': in_qdrant,
            'has_clusters': len(clusters) > 0,
            'cluster_count': len(clusters),
            'case_name': docket.get('case_name', 'No case name'),
            'court': docket.get('court', 'Unknown'),
            'date_created': docket.get('date_created', 'Unknown'),
            'opinions_analysis': []
        }
        
        # Deep dive into clusters and opinions
        if clusters:
            for cluster_idx, cluster_url in enumerate(clusters):
                try:
                    cluster_resp = requests.get(cluster_url, headers=HEADERS, timeout=30)
                    cluster_resp.raise_for_status()
                    cluster = cluster_resp.json()
                    
                    opinions = cluster.get("sub_opinions", [])
                    cluster_analysis = {
                        'cluster_idx': cluster_idx,
                        'has_opinions': len(opinions) > 0,
                        'opinion_count': len(opinions),
                        'case_name': cluster.get('case_name', 'Unknown'),
                        'date_filed': cluster.get('date_filed', 'Unknown'),
                        'opinions_with_text': 0,
                        'total_text_length': 0
                    }
                    
                    # Check each opinion for text content
                    for opinion_url in opinions:
                        try:
                            opinion_resp = requests.get(opinion_url, headers=HEADERS, timeout=30)
                            opinion_resp.raise_for_status()
                            opinion = opinion_resp.json()
                            
                            # Check for text in priority order
                            has_text = False
                            text_length = 0
                            for field in ['html_columbia', 'html_lawbox', 'html_anon_2020', 
                                         'html_with_citations', 'html', 'plain_text']:
                                if opinion.get(field):
                                    text_length = len(str(opinion[field]).strip())
                                    if text_length > 100:  # Minimum useful text
                                        has_text = True
                                        cluster_analysis['opinions_with_text'] += 1
                                        cluster_analysis['total_text_length'] += text_length
                                        break
                            
                            if not has_text:
                                print(f"    ⚠️ Opinion {opinion.get('id')} has no substantial text")
                        
                        except Exception as e:
                            print(f"    ❌ Failed to fetch opinion: {e}")
                            continue
                    
                    docket_analysis['opinions_analysis'].append(cluster_analysis)
                    
                except Exception as e:
                    print(f"  ❌ Failed to fetch cluster: {e}")
                    continue
        
        # Calculate totals
        total_opinions_with_text = sum(c['opinions_with_text'] for c in docket_analysis['opinions_analysis'])
        total_text_length = sum(c['total_text_length'] for c in docket_analysis['opinions_analysis'])
        
        docket_analysis['total_opinions_with_text'] = total_opinions_with_text
        docket_analysis['total_text_length'] = total_text_length
        docket_analysis['has_processable_content'] = total_opinions_with_text > 0 and total_text_length > 500
        
        # Categorize
        if in_qdrant:
            analysis_results['in_qdrant'].append(docket_analysis)
        else:
            analysis_results['not_in_qdrant'].append(docket_analysis)
        
        analysis_results['analysis'].append(docket_analysis)
        
        # Status update
        status = "✅ In Qdrant" if in_qdrant else "❌ Not in Qdrant"
        content_status = "📝 Has content" if docket_analysis['has_processable_content'] else "📭 No content"
        print(f"    {status} | {content_status} | Clusters: {len(clusters)} | Text opinions: {total_opinions_with_text}")
    
    return analysis_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate missing dockets")
    parser.add_argument('--court', default='scotus', help='Court identifier')
    parser.add_argument('--num-dockets', type=int, default=25, help='Number of recent dockets to analyze')
    
    args = parser.parse_args()
    
    results = fetch_and_analyze_recent_dockets(args.court, args.num_dockets)
    
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    in_qdrant_count = len(results['in_qdrant'])
    not_in_qdrant_count = len(results['not_in_qdrant'])
    
    print(f"Dockets in Qdrant: {in_qdrant_count}")
    print(f"Dockets NOT in Qdrant: {not_in_qdrant_count}")
    
    # Analyze the missing ones
    print(f"\n🔍 MISSING DOCKETS ANALYSIS:")
    for docket in results['not_in_qdrant'][:10]:  # Show first 10
        print(f"\nDocket: {docket['docket_number']}")
        print(f"  Case: {docket['case_name']}")
        print(f"  Clusters: {docket['cluster_count']}")
        print(f"  Opinions with text: {docket['total_opinions_with_text']}")
        print(f"  Total text length: {docket['total_text_length']:,} chars")
        print(f"  Processable: {'YES' if docket['has_processable_content'] else 'NO'}")
        
        if not docket['has_processable_content']:
            if docket['cluster_count'] == 0:
                print(f"  ❌ REASON: No clusters found")
            elif docket['total_opinions_with_text'] == 0:
                print(f"  ❌ REASON: No opinions with substantial text")
            elif docket['total_text_length'] < 500:
                print(f"  ❌ REASON: Text too short ({docket['total_text_length']} chars)")
    
    # Summary of reasons
    print(f"\n📋 MISSING DOCKET REASONS:")
    no_clusters = len([d for d in results['not_in_qdrant'] if d['cluster_count'] == 0])
    no_opinions = len([d for d in results['not_in_qdrant'] if d['cluster_count'] > 0 and d['total_opinions_with_text'] == 0])
    text_too_short = len([d for d in results['not_in_qdrant'] if d['total_opinions_with_text'] > 0 and d['total_text_length'] < 500])
    
    print(f"  No clusters: {no_clusters}")
    print(f"  No opinions with text: {no_opinions}")
    print(f"  Text too short: {text_too_short}")
    print(f"  Other reasons: {not_in_qdrant_count - no_clusters - no_opinions - text_too_short}")

if __name__ == "__main__":
    main()