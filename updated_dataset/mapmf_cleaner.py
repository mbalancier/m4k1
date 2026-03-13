"""
MAPMF Data Cleaner
Cleans and processes the scraped mapmf_alerts.csv data
"""

import pandas as pd
import ast
import re
from html import unescape

def parse_list_column(value):
    """Convert string representation of list to actual list or return empty list."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def list_to_pipe_separated(value):
    """Convert list to pipe-separated string for CSV compatibility."""
    if isinstance(value, list):
        return ' | '.join(str(item) for item in value)
    return value

def clean_text(text):
    """Clean text content: remove extra whitespace, decode HTML entities."""
    if pd.isna(text):
        return ''
    # Decode HTML entities
    text = unescape(str(text))
    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def extract_primary_incident_type(types_list):
    """Extract the primary (first) incident type from a list."""
    if isinstance(types_list, list) and len(types_list) > 0:
        return types_list[0]
    return None

def extract_genders_from_subjects(subjects):
    """Extract gender information from the subjects field."""
    if not isinstance(subjects, list):
        return []
    genders = []
    for subject in subjects:
        if isinstance(subject, dict) and 'gender' in subject:
            genders.append(subject['gender'])
    return genders

def count_gender(gender_list, target_gender):
    """Count occurrences of a specific gender in the list."""
    if not isinstance(gender_list, list):
        return 0
    return sum(1 for g in gender_list if g and target_gender.lower() in str(g).lower())

def parse_subjects_column(value):
    """Parse the subjects column which contains list of dicts."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def clean_mapmf_data(input_file='mapmf_alerts.csv', output_file='mapmf_alerts_cleaned.csv'):
    """Main cleaning function."""
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # --- 1. Parse list columns ---
    list_columns = [
        'type_of_incident', 'source_of_incident', 'context_of_incident', 'region_names',
        'gender', 'who_was_attacked', 'type_of_journalist_or_media_actor', 'employment_status'
    ]
    
    for col in list_columns:
        if col in df.columns:
            print(f"Parsing list column: {col}")
            df[col] = df[col].apply(parse_list_column)
    
    # Parse subjects column (list of dicts)
    if 'subjects' in df.columns:
        print("Parsing subjects column...")
        df['subjects'] = df['subjects'].apply(parse_subjects_column)
    
    # --- 2. Clean text columns ---
    text_columns = ['title', 'content']
    
    for col in text_columns:
        if col in df.columns:
            print(f"Cleaning text column: {col}")
            df[col] = df[col].apply(clean_text)
    
    # --- 3. Create useful derived columns ---
    print("Creating derived columns...")
    
    # Primary incident type (first one in the list)
    if 'type_of_incident' in df.columns:
        df['primary_incident_type'] = df['type_of_incident'].apply(extract_primary_incident_type)
        df['incident_type_count'] = df['type_of_incident'].apply(len)
    
    # Primary source of incident
    if 'source_of_incident' in df.columns:
        df['primary_source'] = df['source_of_incident'].apply(extract_primary_incident_type)
    
    # Extract country from region_names if country column is missing/empty
    if 'region_names' in df.columns:
        df['region_level_1'] = df['region_names'].apply(
            lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None
        )
        df['region_level_2'] = df['region_names'].apply(
            lambda x: x[2] if isinstance(x, list) and len(x) > 2 else None
        )
    
    # --- Gender-related derived columns ---
    if 'gender' in df.columns:
        print("Creating gender-related columns...")
        df['primary_gender'] = df['gender'].apply(extract_primary_incident_type)
        df['gender_male_count'] = df['gender'].apply(lambda x: count_gender(x, 'man'))
        df['gender_female_count'] = df['gender'].apply(lambda x: count_gender(x, 'woman'))
        df['gender_other_count'] = df['gender'].apply(
            lambda x: len([g for g in x if isinstance(g, str) and g.lower() not in ['man', 'woman', 'not applicable', '']])
            if isinstance(x, list) else 0
        )
    
    # Extract detailed subject info
    if 'subjects' in df.columns:
        df['subjects_count'] = df['subjects'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        # Extract journalist types from subjects
        df['journalist_types'] = df['subjects'].apply(
            lambda subjects: [s.get('type', '') for s in subjects if isinstance(s, dict) and s.get('kind') == 'journalist']
            if isinstance(subjects, list) else []
        )
    
    # Primary who was attacked
    if 'who_was_attacked' in df.columns:
        df['primary_victim_type'] = df['who_was_attacked'].apply(extract_primary_incident_type)
    
    # --- 4. Clean date column ---
    if 'date' in df.columns:
        print("Parsing dates...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_series = pd.to_datetime(df['date'])
        df['month'] = date_series.dt.month
        df['day_of_week'] = date_series.dt.day_name()
    
    # --- 5. Content length ---
    if 'content' in df.columns:
        df['content_length'] = df['content'].str.len()
    
    # --- 6. Handle missing values ---
    print("Handling missing values...")
    df['country'] = df['country'].fillna('Unknown')
    df['title'] = df['title'].fillna('')
    df['content'] = df['content'].fillna('')
    
    # --- 7. Convert list columns to pipe-separated strings for CSV ---
    print("Converting lists to pipe-separated strings...")
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(list_to_pipe_separated)
    
    # Also convert journalist_types if it exists
    if 'journalist_types' in df.columns:
        df['journalist_types'] = df['journalist_types'].apply(list_to_pipe_separated)
    
    # Convert subjects to JSON string for CSV storage
    if 'subjects' in df.columns:
        import json
        df['subjects'] = df['subjects'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    # --- 8. Reorder columns for better readability ---
    preferred_order = [
        'id', 'title', 'date', 'year', 'month', 'day_of_week',
        'country', 'region_level_1', 'region_level_2',
        'primary_incident_type', 'incident_type_count', 'type_of_incident',
        'primary_source', 'source_of_incident',
        'context_of_incident', 'region_names',
        # Gender and victim information
        'primary_gender', 'gender_male_count', 'gender_female_count', 'gender_other_count', 'gender',
        'primary_victim_type', 'who_was_attacked', 'attacked_count',
        'type_of_journalist_or_media_actor', 'employment_status',
        'subjects_count', 'journalist_types', 'subjects',
        # Content
        'content', 'content_length',
        'published_at_date', '_geo_lat', '_geo_lng'
    ]
    
    # Keep only columns that exist
    final_columns = [col for col in preferred_order if col in df.columns]
    # Add any remaining columns not in preferred order
    remaining = [col for col in df.columns if col not in final_columns]
    final_columns.extend(remaining)
    
    df = df[final_columns]
    
    # --- 9. Remove duplicates ---
    original_count = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    duplicates_removed = original_count - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate entries")
    
    # --- 10. Sort by date (most recent first) ---
    df = df.sort_values('date', ascending=False)
    
    # --- Save cleaned data ---
    # Use comma delimiter, quote only non-numeric fields for Rainbow CSV compatibility
    import csv
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\nCleaned data saved to {output_file}")
    print(f"Final shape: {df.shape}")
    
    # --- Print summary statistics ---
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nTotal alerts: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Countries covered: {df['country'].nunique()}")
    
    print(f"\nTop 10 countries by incidents:")
    print(df['country'].value_counts().head(10).to_string())
    
    if 'primary_incident_type' in df.columns:
        print(f"\nTop 10 incident types:")
        print(df['primary_incident_type'].value_counts().head(10).to_string())
    
    if 'year' in df.columns:
        print(f"\nIncidents by year:")
        print(df['year'].value_counts().sort_index().to_string())
    
    if 'primary_gender' in df.columns:
        print(f"\nGender distribution:")
        print(df['primary_gender'].value_counts().to_string())
    
    if 'gender_male_count' in df.columns and 'gender_female_count' in df.columns:
        total_male = df['gender_male_count'].sum()
        total_female = df['gender_female_count'].sum()
        print(f"\nTotal male victims: {total_male}")
        print(f"Total female victims: {total_female}")
    
    return df


if __name__ == "__main__":
    df = clean_mapmf_data()
    
    # print("\n" + "-"*50)
    # print("SAMPLE OF CLEANED DATA")
    # print("-"*50)
    print(df[['id', 'title', 'date', 'country', 'primary_incident_type']].head(10).to_string())
