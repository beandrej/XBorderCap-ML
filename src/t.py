import pandas as pd
import json
import os
import config
# These should match your config
CLS_COLS = [
    'GBR_BEL', 'BEL_GBR', 'ITA_SVN', 'NET_DK1', 'ITA_FRA', 'ITA_SWI', 'GER_SWI', 
    'ITA_AUS', 'FRA_SWI', 'SWI_GER', 'AUS_SWI', 'GBR_NET', 'NET_GBR', 'SWI_FRA',
    'GBR_FRA', 'SWI_AUS', 'FRA_GBR', 'NO2_NET', 'GER_DK2', 'DK1_NET', 'NET_NO2',
    'DK2_GER'
]

def generate_class_mapping(csv_path, output_path="mappings/class_mapping.json"):
    df = pd.read_csv(csv_path)

    class_mapping = {}

    for col in CLS_COLS:
        if col not in df.columns:
            print(f"⚠️ Skipping {col} — not found in data.")
            continue

        unique_vals = sorted(df[col].dropna().unique())
        mapping = {str(val): idx for idx, val in enumerate(unique_vals)}
        class_mapping[col] = mapping

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)

    print(f"✅ Class mapping saved to: {output_path}")

if __name__ == "__main__":
    #file_path = 'C:/Users/andre/github/XBorderCap-ML/prep_data/targets/NTC_normed.csv'
    classmap_path = os.path.join(config.PROJECT_ROOT, "mappings", f"clsMap_.json")
    print(classmap_path)
    #generate_class_mapping(file_path)