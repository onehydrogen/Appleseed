import pandas as pd
try:
    df = pd.read_csv('C:/Users/bendw/Downloads/Appleseed-main/Appleseed-main/MyCSVApp/data/legislative_analysis_AR_2025_20250217_111047.csv')
    print(df.head())
except Exception as e:
    print(f"Error: {e}")