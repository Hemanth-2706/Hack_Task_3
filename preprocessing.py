import pandas as pd
from tqdm import tqdm

def load_data(data_path="."):
    """
    Loads all necessary Synthea CSV files into pandas DataFrames.
    Assumes CSV files are in the specified directory.
    """
    try:
        patients = pd.read_csv(f"{data_path}/patients.csv")
        encounters = pd.read_csv(f"{data_path}/encounters.csv")
        conditions = pd.read_csv(f"{data_path}/conditions.csv")
        medications = pd.read_csv(f"{data_path}/medications.csv")
        observations = pd.read_csv(f"{data_path}/observations.csv")
        procedures = pd.read_csv(f"{data_path}/procedures.csv")
        
        print("All CSV files loaded successfully.")
        
        return {
            "patients": patients,
            "encounters": encounters,
            "conditions": conditions,
            "medications": medications,
            "observations": observations,
            "procedures": procedures,
        }
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure your CSV files are in the '{data_path}' directory.")
        return None

def filter_columns(dataframes):
    """
    Selects only the medically relevant columns and ignores sensitive or financial data.
    """
    # Columns to keep for each table
    patients_cols = ['Id', 'BIRTHDATE', 'RACE', 'ETHNICITY', 'GENDER']
    encounters_cols = ['Id', 'START', 'STOP', 'PATIENT', 'ENCOUNTERCLASS', 'CODE', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION']
    conditions_cols = ['START', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION']
    medications_cols = ['START', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION']
    observations_cols = ['DATE', 'PATIENT', 'ENCOUNTER', 'CATEGORY', 'CODE', 'DESCRIPTION', 'VALUE', 'UNITS']
    procedures_cols = ['START', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION']

    dataframes["patients"] = dataframes["patients"][patients_cols]
    dataframes["encounters"] = dataframes["encounters"][encounters_cols]
    dataframes["conditions"] = dataframes["conditions"][conditions_cols]
    dataframes["medications"] = dataframes["medications"][medications_cols]
    dataframes["observations"] = dataframes["observations"][observations_cols]
    dataframes["procedures"] = dataframes["procedures"][procedures_cols]
    
    print("Filtered out irrelevant columns.")
    return dataframes

def generate_evidence_snippets(dataframes):
    """
    Generates natural language snippets from the structured data.
    """
    # Create a mapping from patient ID to gender for quick lookup
    patient_gender_map = dataframes["patients"].set_index('Id')['GENDER'].to_dict()
    
    snippets = []

    def get_full_gender(patient_id):
        """Helper function to convert gender abbreviation to full word."""
        gender_abbr = patient_gender_map.get(patient_id, 'patient').lower()
        if gender_abbr == 'm':
            return 'male'
        elif gender_abbr == 'f':
            return 'female'
        return 'patient' # Fallback for unknown genders
    
    # Process Conditions
    for _, row in tqdm(dataframes["conditions"].iterrows(), total=len(dataframes["conditions"]), desc="Processing Conditions"):
        gender = get_full_gender(row['PATIENT'])
        text = f"This {gender} patient has a diagnosis of '{row['DESCRIPTION']}' (ICD-10 Code: {row['CODE']}), which started on {row['START'].split('T')[0]}."
        snippets.append({
            "patient_id": row['PATIENT'],
            "encounter_id": row['ENCOUNTER'],
            "gender": gender.capitalize(),
            "source": "conditions",
            "text": text
        })
        
    # Process Medications
    for _, row in tqdm(dataframes["medications"].iterrows(), total=len(dataframes["medications"]), desc="Processing Medications"):
        gender = get_full_gender(row['PATIENT'])
        reason = f" for reason: '{row['REASONDESCRIPTION']}' (Code: {row['REASONCODE']})" if pd.notna(row['REASONCODE']) else ""
        text = f"This {gender} patient was prescribed '{row['DESCRIPTION']}' (RxNorm Code: {row['CODE']}) starting on {row['START'].split('T')[0]}{reason}."
        snippets.append({
            "patient_id": row['PATIENT'],
            "encounter_id": row['ENCOUNTER'],
            "gender": gender.capitalize(),
            "source": "medications",
            "text": text
        })

    # Process Observations (Labs and Vitals)
    for _, row in tqdm(dataframes["observations"].iterrows(), total=len(dataframes["observations"]), desc="Processing Observations"):
        gender = get_full_gender(row['PATIENT'])
        # Clean up value if it's a string that can be float
        value_str = row['VALUE']
        try:
            value_str = f"{float(row['VALUE']):.2f}"
        except (ValueError, TypeError):
            pass # Keep as string if conversion fails
            
        text = f"This {gender} patient had an observation on {row['DATE'].split('T')[0]} of type '{row['CATEGORY']}': '{row['DESCRIPTION']}' (LOINC Code: {row['CODE']}) with a value of {value_str} {row['UNITS']}."
        snippets.append({
            "patient_id": row['PATIENT'],
            "encounter_id": row['ENCOUNTER'],
            "gender": gender.capitalize(),
            "source": "observations",
            "text": text
        })
        
    # Process Procedures
    for _, row in tqdm(dataframes["procedures"].iterrows(), total=len(dataframes["procedures"]), desc="Processing Procedures"):
        gender = get_full_gender(row['PATIENT'])
        reason = f" for reason: '{row['REASONDESCRIPTION']}' (Code: {row['REASONCODE']})" if pd.notna(row['REASONCODE']) else ""
        text = f"This {gender} patient underwent the procedure '{row['DESCRIPTION']}' (SNOMED-CT Code: {row['CODE']}) on {row['START'].split('T')[0]}{reason}."
        snippets.append({
            "patient_id": row['PATIENT'],
            "encounter_id": row['ENCOUNTER'],
            "gender": gender.capitalize(),
            "source": "procedures",
            "text": text
        })
        
    print(f"\nGenerated a total of {len(snippets)} evidence snippets.")
    return pd.DataFrame(snippets)


if __name__ == "__main__":
    # Specify the path to your Synthea CSV files
    SYNTHEA_DATA_PATH = "./csv" # IMPORTANT: Change this to your data directory
    
    # 1. Load data
    all_data = load_data(SYNTHEA_DATA_PATH)
    
    if all_data:
        # 2. Filter columns to keep only relevant ones
        filtered_data = filter_columns(all_data)
        
        # 3. Generate text-based evidence snippets
        snippets_df = generate_evidence_snippets(filtered_data)
        
        # 4. Save the final corpus for the next stage (indexing)
        output_path = "evidence_snippets.parquet"
        snippets_df.to_parquet(output_path, index=False)
        
        print(f"\nPreprocessing complete. Evidence snippets saved to '{output_path}'.")
        print("This file is now ready for the indexing stage (BM25 and FAISS with ClinicalBERT).")

