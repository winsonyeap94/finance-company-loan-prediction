import logging
import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# ============================== Auxiliary Function ==============================
def _clean_colnames(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df.columns = [re.sub('_+', '_', re.sub(r'(?<!^)(?=[A-Z])', '_', x).upper()).replace('I_D', 'ID') for x in data_df.columns]
    return data_df


def _preprocess_numerical_data(data_df: pd.DataFrame, reference_df: pd.DataFrame=None) -> pd.DataFrame:
    numeric_vars = [
        'APPLICANT_INCOME', 'COAPPLICANT_INCOME', 'LOAN_AMOUNT', 'LOAN_AMOUNT_TERM',
    ]
    for var in numeric_vars:
        data_df[var] = pd.to_numeric(data_df[var], errors='coerce')
    return data_df


def _preprocess_categorical_data(data_df: pd.DataFrame, reference_df: pd.DataFrame=None) -> pd.DataFrame:
    data_df['GENDER'] = data_df['GENDER'].fillna('MISSING')
    data_df['LOAN_ID'] = data_df['LOAN_ID'].astype(str)
    data_df['MARRIED'] = data_df['MARRIED'].fillna('No')
    data_df['SELF_EMPLOYED'] = data_df['SELF_EMPLOYED'].fillna('No')
    data_df['CREDIT_HISTORY'] = data_df['CREDIT_HISTORY'].fillna(0)
    data_df['DEPENDENTS'] = data_df['DEPENDENTS'].fillna('0')
    if reference_df is None:
        reference_df = data_df.copy()
    data_df['GENDER'] = pd.Categorical(data_df['GENDER'], categories=['Male', 'Female', 'MISSING'], ordered=False)
    data_df['MARRIED'] = pd.Categorical(data_df['MARRIED'], categories=['No', 'Yes'], ordered=False)
    data_df['DEPENDENTS'] = pd.Categorical(data_df['DEPENDENTS'], categories=['0', '1', '2', '3+'], ordered=True)
    data_df['EDUCATION'] = pd.Categorical(data_df['EDUCATION'], categories=['Not Graduate', 'Graduate'], ordered=False)
    data_df['SELF_EMPLOYED'] = pd.Categorical(data_df['SELF_EMPLOYED'], categories=['No', 'Yes'], ordered=False)
    data_df['PROPERTY_AREA'] = pd.Categorical(data_df['PROPERTY_AREA'], categories=['Rural', 'Semiurban', 'Urban'], ordered=True)
    if 'LOAN_STATUS' in data_df.columns:
        data_df['LOAN_STATUS'] = np.where(data_df['LOAN_STATUS'].str.upper() == 'Y', 1, 0)
    return data_df


def _impute_data(data_df: pd.DataFrame, imputer=None) -> pd.DataFrame:
    # Setting up imputation pipeline
    impute_numeric_cols = [
        'APPLICANT_INCOME', 'COAPPLICANT_INCOME', 'LOAN_AMOUNT', 'LOAN_AMOUNT_TERM', 'CREDIT_HISTORY'
    ]
    impute_categorical_cols = [
        'GENDER', 'MARRIED', 'DEPENDENTS', 'EDUCATION', 'SELF_EMPLOYED', 'PROPERTY_AREA'
    ]
    impute_ordinal_cols = [x for x in impute_categorical_cols if data_df[x].dtype.ordered]
    impute_nominal_cols = [x for x in impute_categorical_cols if not data_df[x].dtype.ordered]
    impute_cols = impute_numeric_cols + impute_ordinal_cols + impute_nominal_cols
    _logger = logging.getLogger(__name__)
    _logger.debug(f"Numeric Vars for Imputation: {impute_numeric_cols}")
    _logger.debug(f"Ordinal Vars for Imputation: {impute_ordinal_cols}")
    _logger.debug(f"Nominal Vars for Imputation: {impute_nominal_cols}")
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    ordinal_transformer = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder())
    ])
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('numeric', numerical_transformer, impute_numeric_cols),
            ('ordinal', ordinal_transformer, impute_ordinal_cols),
            ('nominal', nominal_transformer, impute_nominal_cols)
        ]
    )
    imputer_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('imputer', KNNImputer(n_neighbors=5))
    ])
    
    # Running Imputation
    pre_imputation_df = data_df.copy()
    transformed_df = imputer_pipeline.fit_transform(data_df[impute_cols])
    transformed_columns = preprocessor_pipeline.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_df, columns=transformed_columns, index=data_df.index)
    _logger.debug(f"NaNs before imputation: {data_df[impute_cols].isna().sum()}")
    for var in impute_numeric_cols:
        data_df[var] = data_df[var].fillna(transformed_df[f'numeric__{var}'])
    _logger.debug(f"NaNs after imputation: {data_df[impute_cols].isna().sum()}")
    
    # Ensuring values do not go out of bounds
    data_df['LOAN_AMOUNT'] = data_df['LOAN_AMOUNT'].clip(lower=pre_imputation_df['LOAN_AMOUNT'].min())
    data_df['LOAN_AMOUNT_TERM'] = data_df['LOAN_AMOUNT_TERM'].clip(lower=pre_imputation_df['LOAN_AMOUNT_TERM'].min())
    
    return data_df, imputer_pipeline


# ============================== Main Function ==============================
def preprocess_dataset(data_df: pd.DataFrame, reference_df: pd.DataFrame=None, imputer=None) -> pd.DataFrame:
    data_df = _clean_colnames(data_df)
    data_df = _preprocess_numerical_data(data_df, reference_df)
    data_df = _preprocess_categorical_data(data_df, reference_df)
    data_df, imputer = _impute_data(data_df, imputer)
    return data_df, imputer
