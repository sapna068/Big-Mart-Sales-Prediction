
import argparse
import json
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def log(msg: str):
    print(msg, flush=True)


def normalize_fat_content(s: pd.Series) -> pd.Series:
    mapping = {
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular',
        'Regular': 'Regular',
        'Low Fat': 'Low Fat'
    }
    s_clean = s.fillna('Unknown').astype(str).str.strip()
    s_norm = s_clean.str.lower().map(mapping).fillna(s_clean.str.title())
    
    s_norm = s_norm.replace({'Non Edible': 'Non-Edible', 'Non Edible ': 'Non-Edible'})
    return s_norm


def map_item_category_from_item_type(df: pd.DataFrame) -> pd.Series:
    """
    Build Item_Category using business rules on Item_Type and Fat Content:
      - Drinks: Item_Type in {"Soft Drinks","Hard Drinks"}
      - Non-Consumable: Item_Fat_Content == "Non-Edible"
      - Food: everything else (including Dairy, Bakery, Meat, etc.)
    Modify the mapping below to suit your project.
    """
    drinks = {'Soft Drinks', 'Hard Drinks'}

    # Start with Food
    cat = pd.Series('Food', index=df.index)

    # Drinks by Item_Type
    cat[df['Item_Type'].astype(str).isin(drinks)] = 'Drinks'

    # Override to Non-Consumable if explicitly flagged by fat-content
    fat = df['Item_Fat_Content'].astype(str)
    cat[fat.eq('Non-Edible')] = 'Non-Consumable'

    # Example: ensure "Dairy" maps to Food (per your last request)
    # Already covered by default 'Food' assignment.

    return cat.astype(str)


def add_engineered_columns(df: pd.DataFrame, reference_year: int = 2013) -> pd.DataFrame:
  
    out = df.copy()

    # Item_Visibility_Log
    if 'Item_Visibility' in out.columns and 'Item_Visibility_Log' not in out.columns:
        out['Item_Visibility_Log'] = np.log1p(out['Item_Visibility'].astype(float))

    # Outlet_Age
    if 'Outlet_Establishment_Year' in out.columns and 'Outlet_Age' not in out.columns:
        out['Outlet_Age'] = (reference_year - out['Outlet_Establishment_Year'].astype(float)).clip(lower=0)

    # Clean/normalize fat content
    if 'Item_Fat_Content' in out.columns:
        out['Item_Fat_Content'] = normalize_fat_content(out['Item_Fat_Content'])

    # Item_Category (Food / Non-Consumable / Drinks) from Item_Type and Fat
    if 'Item_Category' not in out.columns and 'Item_Type' in out.columns:
        out['Item_Category'] = map_item_category_from_item_type(out)

    return out


def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    
    df_ohe = pd.get_dummies(df, columns=[c for c in cols if c in df.columns], drop_first=False, dtype=np.uint8)
    return df_ohe


def align_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    for c in sorted(train_cols - test_cols):
        test_df[c] = 0
    for c in sorted(test_cols - train_cols):
        train_df[c] = 0

    # Reorder to the same column order
    test_df = test_df[train_df.columns.tolist()]
    return train_df, test_df


def select_model_features(df: pd.DataFrame) -> List[str]:
    
    core = [
        'Item_Weight','Item_Visibility','Item_MRP',
        'Outlet_Establishment_Year','Outlet_Age','Item_Visibility_Log'
    ]

    
    ohe_wanted = [
        'Outlet_Type_Supermarket Type1','Outlet_Type_Supermarket Type2','Outlet_Type_Supermarket Type3',
        'Outlet_Size_Medium','Outlet_Size_Small',
        'Item_Fat_Content_Non-Edible','Item_Fat_Content_Regular',
        'Item_Category_Food','Item_Category_Non-Consumable','Item_Category_Drinks'
    ]

    features = [c for c in core + ohe_wanted if c in df.columns]
    return features


def run_kfold_xgb(
    X: pd.DataFrame, y: np.ndarray, params: dict,
    n_splits: int = 5, early_stopping_rounds: int = 100, num_boost_round: int = 5000
) -> Tuple[float, float, int, np.ndarray]:
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=float)
    rmses = []
    best_rounds = []

    X_values = X.values  # speed
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_values), 1):
        dtr  = xgb.DMatrix(X_values[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X_values[val_idx], label=y[val_idx])

        bst = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr,'train'), (dval,'valid')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        # Predict with best iteration (version-safe)
        try:
            preds_val = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
            best_iter = int(getattr(bst, 'best_iteration', 0))
        except Exception:
            best_iter = int(getattr(bst, 'best_iteration', 0))
            preds_val = bst.predict(dval, iteration_range=(0, best_iter + 1))

        oof[val_idx] = preds_val
        rmse = float(np.sqrt(mean_squared_error(y[val_idx], preds_val)))
        rmses.append(rmse)
        best_rounds.append(best_iter if best_iter > 0 else len(bst.get_fscore()))

        log(f"[Fold {fold}] RMSE={rmse:.3f}, best_iter≈{best_rounds[-1]}")

    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    final_rounds = int(np.median(best_rounds) + 50)  # small cushion
    return mean_rmse, std_rmse, final_rounds, oof


def train_final_xgb(X: pd.DataFrame, y: np.ndarray, params: dict, num_rounds: int) -> xgb.Booster:
    dtrain = xgb.DMatrix(X.values, label=y)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain,'train')],
        verbose_eval=False
    )
    return bst



def main(args):
    # Read data
    train = pd.read_csv(args.train_csv)
    test  = pd.read_csv(args.test_csv)

    # Ensure target exists in train
    if 'Item_Outlet_Sales' not in train.columns:
        raise ValueError("Train CSV must include 'Item_Outlet_Sales'.")

    # Basic cleaning of missing numeric columns (median impute for stability)
    for col in ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median())
        if col in test.columns:
            test[col] = test[col].fillna(train[col].median())  # use train stats

    
    train = add_engineered_columns(train, reference_year=args.reference_year)
    test  = add_engineered_columns(test,  reference_year=args.reference_year)

   
    ohe_cols = ['Outlet_Type', 'Outlet_Size', 'Item_Fat_Content', 'Item_Category']
    train_ohe = one_hot_encode(train, ohe_cols)
    test_ohe  = one_hot_encode(test,  ohe_cols)

    
    train_ohe, test_ohe = align_train_test(train_ohe, test_ohe)

    
    feature_cols = select_model_features(train_ohe)

  
    X = train_ohe[feature_cols].copy()
    y = train_ohe['Item_Outlet_Sales'].astype(float).values
    X_test = test_ohe[feature_cols].copy()

    
    non_num_train = X.select_dtypes(include=['object','category']).columns.tolist()
    non_num_test  = X_test.select_dtypes(include=['object','category']).columns.tolist()
    if non_num_train or non_num_test:
        raise TypeError(f"Non-numeric columns present. Train: {non_num_train} | Test: {non_num_test}")

    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'seed': 42
    }

    log("Running 5-fold CV with early stopping...")
    mean_rmse, std_rmse, final_rounds, oof = run_kfold_xgb(
        X, y, params, n_splits=5, early_stopping_rounds=100, num_boost_round=5000
    )
    log(f"CV RMSE: mean={mean_rmse:.3f} | std={std_rmse:.3f} | final_rounds={final_rounds}")

    
    log("Training final model on all training data...")
    bst_final = train_final_xgb(X, y, params, num_rounds=final_rounds)

    
    if args.model_path:
        bst_final.save_model(args.model_path)
        log(f"Saved model: {args.model_path}")

    # Predict on test
    dtest = xgb.DMatrix(X_test.values)
    try:
        preds = bst_final.predict(dtest, ntree_limit=getattr(bst_final, 'best_ntree_limit', 0))
        if preds is None or len(preds) != len(X_test):
            preds = bst_final.predict(dtest)
    except Exception:
        preds = bst_final.predict(dtest)

    preds = np.clip(preds, 0, None)

   
    id_cols = [c for c in ['Item_Identifier','Outlet_Identifier'] if c in test.columns]
    if not id_cols:
        
        id_cols = []
        log("Warning: 'Item_Identifier'/'Outlet_Identifier' not found in test; submission will contain only predictions.")

    submission = test[id_cols].copy() if id_cols else pd.DataFrame(index=test.index)
    submission['Item_Outlet_Sales'] = preds
    submission.to_csv(args.submission_csv, index=False)
    log(f"Saved submission: {args.submission_csv}")

    
    baseline_rmse = float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean()))))
    model_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    improvement = (1 - model_rmse / baseline_rmse) * 100.0
    log(f"Baseline (global mean) RMSE={baseline_rmse:.2f} | OOF RMSE={model_rmse:.2f} | Improvement={improvement:.1f}%")

    
    if args.report_json:
        report = {
            "cv_rmse_mean": mean_rmse,
            "cv_rmse_std": std_rmse,
            "oof_rmse": model_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_pct": improvement,
            "final_rounds": final_rounds,
            "n_features": len(feature_cols),
            "features": feature_cols
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"Saved report: {args.report_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigMart XGBoost pipeline")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV with Item_Outlet_Sales")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--submission_csv", type=str, default="submission_xgb.csv", help="Output submission CSV path")
    parser.add_argument("--model_path", type=str, default="xgb_model.json", help="Path to save XGBoost model (.json)")
    parser.add_argument("--report_json", type=str, default="report.json", help="Optional metrics report JSON")
    parser.add_argument("--reference_year", type=int, default=2013, help="Reference year for Outlet_Age feature")
    args = parser.parse_args()
    main(args)
