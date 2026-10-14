import pickle
import glob
import os

files = glob.glob('/Users/tanayasalunke/Downloads/SMA-NLP-Project--master/final_charts/classification_results*.pkl')
files.append('/Users/tanayasalunke/Downloads/SMA-NLP-Project--master/classification_results_xgboost.pkl')

for file in files:
    if os.path.exists(file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                filename = os.path.basename(file)
                print(f"\n--- {filename} ---")
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict) and 'accuracy' in v:
                            print(f"{k}: {v['accuracy']:.4f}")
                        elif isinstance(v, float):
                            print(f"{k}: {v:.4f}")
                        else:
                            pass # skip large dicts
                else:
                    print("Data is not a dict.")
        except Exception as e:
            print(f"Error: {e}")
