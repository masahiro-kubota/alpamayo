
import physical_ai_av
import pandas as pd

# Set display options to ensure full IDs are shown
pd.set_option('display.max_colwidth', None)

try:
    print("Initializing Dataset Interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print("Total clips:", len(avdi.clip_index))
    
    # Check for specific ID
    target_short = 'f789b390'
    print(f"\nChecking for {target_short}...")
    
    if target_short in avdi.clip_index.index:
        print(f"SUCCESS: {target_short} found directly.")
    else:
        print(f"FAILURE: {target_short} NOT found directly.")
        
        # Search for partial match
        # The index might be full UUIDs
        all_ids = avdi.clip_index.index.astype(str)
        matches = [c for c in all_ids if c.startswith(target_short)]
        
        if matches:
            print(f"Found {len(matches)} matches starting with {target_short}:")
            for m in matches:
                print(f" - {m}")
        else:
            print("No matches found starting with that prefix.")
            print("First 5 IDs in index:")
            print(all_ids[:5].tolist())

except Exception as e:
    print(f"An error occurred: {e}")
