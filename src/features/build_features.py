import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier

def create_model_pipeline():
    df = pd.read_parquet("multisim_dataset.parquet")
    
    ob_to_num = ['age', 'age_dev', 'dev_num', 'is_dualsim', 'is_smartphone']
    for col in ob_to_num:
        df[col] = pd.to_numeric(df[col])
    df.loc[df['age'] == 1941, 'age'] = 2025 - 1941
    df = df.drop("is_featurephone", axis=1)

    X = df.drop('target', axis=1)
    y = df['target']

    exclude_cols = ['is_dualsim', 'is_smartphone']
    numeric_features = X.select_dtypes(include='number').columns.difference(exclude_cols)
    
    few_null_cols = ['dev_man', 'device_os_name', 'region']
    many_null_cols = ['simcard_type']
    
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ("scaler", RobustScaler())
    ])

    few_null_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ('catboost_enc', CatBoostEncoder())
    ])

    many_null_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('catboost_enc', CatBoostEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ('few_null', few_null_pipeline, few_null_cols),
        ('many_null', many_null_pipeline, many_null_cols),
    ])

    model_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            reg_alpha=0.5,
            reg_lambda=1.0
        ))
    ])
    return model_pipeline

def main():
    pipeline = create_model_pipeline()

if __name__ == "__main__":
    main()
