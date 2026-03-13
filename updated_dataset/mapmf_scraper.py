"""
MAPMF Scraper - Using MeiliSearch API
The website renders content with JavaScript, so we use their public search API directly.
"""

import requests
import pandas as pd
import json
import time

# MeiliSearch API credentials (public, found in page source)
MEILI_URL = 'https://www.mapmf.org/meili/'
MEILI_KEY = 'c129ca42527c52965c80099ab1a869f40de8ec3b698d1e361b0cf7402c6d48a1'
MEILI_INDEX = 'alerts'

def fetch_alerts(offset=0, limit=1000):
    """Fetch alerts from MeiliSearch API."""
    headers = {
        'Authorization': f'Bearer {MEILI_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'q': '',  # Empty query returns all
        'limit': limit,
        'offset': offset,
        'sort': ['published_at:desc']  # Most recent first
    }
    
    try:
        response = requests.post(
            f'{MEILI_URL}indexes/{MEILI_INDEX}/search',
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching alerts: {e}")
        return None

def scrape_all_alerts(target_count=11393, output_file='mapmf_alerts.csv'):
    """Scrape all alerts using the MeiliSearch API."""
    
    all_alerts = []
    offset = 0
    batch_size = 1000  # MeiliSearch max limit per request
    
    print(f"Starting scrape. Target: {target_count} alerts")
    
    while len(all_alerts) < target_count:
        print(f"Fetching alerts {offset} to {offset + batch_size}...", end='\r')
        
        result = fetch_alerts(offset=offset, limit=batch_size)
        
        if not result or 'hits' not in result:
            print(f"\nFailed to fetch at offset {offset}")
            break
        
        hits = result['hits']
        if not hits:
            print(f"\nNo more alerts found at offset {offset}")
            break
        
        all_alerts.extend(hits)
        print(f"Collected {len(all_alerts)} alerts so far...")
        
        # Check if we've reached the end
        estimated_total = result.get('estimatedTotalHits', 0)
        if offset + batch_size >= estimated_total:
            print(f"\nReached end of available alerts ({estimated_total} total)")
            break
        
        offset += batch_size
        time.sleep(0.5)  # Be respectful to the server
    
    # Convert to DataFrame with selected columns
    print(f"\nProcessing {len(all_alerts)} alerts...")
    
    df = pd.DataFrame(all_alerts)
    
    # Select and rename important columns
    columns_to_keep = [
        'id', 'title', 'content', 'date', 'country', 
        'type_of_incident', 'source_of_incident', 'context_of_incident',
        'region_names', 'year', 'published_at_date',
        '_geo_lat', '_geo_lng',
        # Journalist/victim information
        'gender', 'subjects', 'who_was_attacked', 
        'type_of_journalist_or_media_actor', 'employment_status',
        'attacked_count'
    ]
    
    # Keep only columns that exist
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nScraping complete! Collected {len(df)} alerts.")
    print(f"Data saved to {output_file}")
    
    return df

def get_total_count():
    """Get the total number of alerts available."""
    result = fetch_alerts(offset=0, limit=1)
    if result:
        return result.get('estimatedTotalHits', 0)
    return 0

if __name__ == "__main__":
    # First check how many alerts are available
    total = get_total_count()
    print(f"Total alerts available: {total}")
    
    # Scrape all available alerts
    df = scrape_all_alerts(target_count=total)
    
    print("\nSample of collected data:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
