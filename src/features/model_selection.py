import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform

def load_and_prepare_data(file_path="multisim_dataset.parquet"):
    df = pd.read_parquet(file_path)
    
    ob_to_num = ['age', 'age_dev', 'dev_num', 'is_dualsim', 'is_smartphone']
    for col in ob_to_num:
        df[col] = pd.to_numeric(df[col])
    df.loc[df['age'] == 1941, 'age'] = 2025 - 1941
    df = df.drop("is_featurephone", axis=1)

    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def select_best_model(pipeline, X_train, y_train):
    param_dist = {
        'classifier__n_estimators': randint(50, 200),
        'classifier__max_depth': randint(3, 8),
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__reg_alpha': uniform(0.0, 1.0),
        'classifier__reg_lambda': uniform(0.0, 1.0)
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    
    print("Best hyperparameters:", random_search.best_params_)
    print(f"Best CV accuracy: {random_search.best_score_:.4f}")

    return random_search.best_estimator_
