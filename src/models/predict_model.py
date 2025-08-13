import pandas as pd
import joblib

def main():
    model = joblib.load("models/best_model.pkl")
    
    df_new = pd.read_parquet("multisim_dataset.parquet")  
    
    predictions = model.predict(df_new)
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()