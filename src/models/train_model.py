import pandas as pd
import joblib
from features.pipeline import create_model_pipeline
from features.model_selection import select_best_model
from sklearn.model_selection import train_test_split
def main():
    df = pd.read_parquet("multisim_dataset.parquet")
    
    ob_to_num = ['age', 'age_dev', 'dev_num', 'is_dualsim', 'is_featurephone', 'is_smartphone']
    for col in ob_to_num:
        df[col] = pd.to_numeric(df[col])
    df.loc[df['age'] == 1941, 'age'] = 2025 - 1941
    df = df.drop("is_featurephone", axis=1)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_model_pipeline()
    
    best_model = select_best_model(pipeline, X_train, y_train)

    joblib.dump(best_model, "models/best_model.pkl")
    print("Best model saved.")
if __name__ == "__main__":
    main()