import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from scipy.stats import pearsonr, chisquare
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset():
    dataset_path = input("Please enter the path of your dataset: ")
    df = pd.read_csv(dataset_path)

    # Extracting numerical and categorical columns
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions',
               'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']

    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Weak Foot']

    # Subsetting columns based on numcols and catcols
    df = df[numcols + catcols]

    # Concatenating original numerical columns and one-hot encoded categorical columns
    df = pd.concat([df[numcols], pd.get_dummies(df[catcols])], axis=1)

    # Extracting feature columns
    features = df.columns

    # Dropping rows with missing values
    df = df.dropna()

    # Creating a DataFrame with columns specified by the features list
    df = pd.DataFrame(df, columns=features)

    # Creating the target variable (y) based on a condition
    y = df['Overall'] >= 87

    # Creating the feature matrix (X) by copying the DataFrame and removing the 'Overall' column
    X = df.copy()
    del X['Overall']

    num_feats = input("How many features do you want set for selections: ")

    return X, y, num_feats



def cor_selector(X, y, num_feats):
    # Create a DataFrame combining X and y
    df = pd.concat([X, y], axis=1)

    # Calculate the Pearson correlation coefficients for each feature
    corr_values = df.corr().iloc[:-1, -1]

    # Sort the features based on their absolute correlation with the target variable
    sorted_features = corr_values.abs().sort_values(ascending=False).index

    # Ensure num_feats is an integer
    num_feats = int(num_feats)

    # Select the top num_feats features
    selected_features = sorted_features[:num_feats]

   
    for feature in selected_features:
        corr = corr_values[feature]
        print(f"{feature}: {corr:.2f}")

    # Define the correlation support and feature
    cor_support = [True if feature in selected_features else False for feature in X.columns]
    cor_feature = selected_features

    # Return the correlation support and feature
    return cor_support, cor_feature



def chi_squared_selector(X, y, num_feats):
    # Use SelectKBest to apply the chi-squared test
    num_feats = int(num_feats)
    chi2_selector = SelectKBest(chi2, k=num_feats)
    chi2_selector.fit(X, y)

    # Get the selected features
    chi_support = chi2_selector.get_support()
    chi_feature = X.columns[chi_support].tolist()


    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    # Scale the features using MinMaxScaler
    num_feats= int(num_feats)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose Logistic Regression as the estimator
    estimator = LogisticRegression()

    # Initialize RFE
    rfe = RFE(estimator, n_features_to_select=num_feats)

    # Fit RFE and get the selected features
    rfe.fit(X_scaled, y)
    rfe_support = rfe.support_
    rfe_feature = X.columns[rfe_support].tolist()

   
    # Your code ends here
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Standardize features using StandardScaler
    print("total number of features set by us is :",num_feats)
    num_feats=int(num_feats)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose Logistic Regression with L1 penalty (Lasso) as the estimator
    estimator = LogisticRegression(penalty='l1', solver='liblinear')

    # Use SelectFromModel to perform feature selection
    embedded_lr_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_lr_selector.fit(X_scaled, y)

    # Get the selected features
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.columns[embedded_lr_support].tolist()

   

    return embedded_lr_support, embedded_lr_feature
    

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    
    # Choosing RandomForestClassifier as the estimator
    estimator = RandomForestClassifier()
    num_feats=int(num_feats)
    # Using SelectFromModel to perform feature selection
    embedded_rf_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_rf_selector.fit(X, y)

    # Get the selected features
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.columns[embedded_rf_support].tolist()

    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Choosing LGBMClassifier as the estimator
    estimator = LGBMClassifier()
    num_feats=int(num_feats)
    # Using SelectFromModel to perform feature selection
    embedded_lgbm_selector = SelectFromModel(estimator, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)

    # Get the selected features
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.columns[embedded_lgbm_support].tolist()

   

    return embedded_lgbm_support, embedded_lgbm_feature

def autoFeatureSelector(methods=[]):
    X, y, num_feats = preprocess_dataset()

    cor_support, cor_feature = cor_selector(X, y, num_feats) if 'pearson' in methods else ([], [])
    chi_support, chi_feature = chi_squared_selector(X, y, num_feats) if 'chi-square' in methods else ([], [])
    rfe_support, rfe_feature = rfe_selector(X, y, num_feats) if 'rfe' in methods else ([], [])
    embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats) if 'log-reg' in methods else ([], [])
    embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats) if 'rf' in methods else ([], [])
    embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats) if 'lgbm' in methods else ([], [])

    # Combine all the above feature lists
    all_features = {
        'Pearson': cor_feature,
        'Chi-Square': chi_feature,
        'RFE': rfe_feature,
        'Logistic Regression': embedded_lr_feature,
        'Random Forest': embedded_rf_feature,
        'LightGBM': embedded_lgbm_feature
    }

    # Count the votes for each feature
    votes = {}
    for method, features in all_features.items():
        for feature in features:
            votes[feature] = votes.get(feature, 0) + 1

    # Find the features with the maximum votes
    max_votes = max(votes.values())
    best_features = [feature for feature, count in votes.items() if count == max_votes]

    return best_features

# Example usage
best_features = autoFeatureSelector(methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print("Best Features:", best_features)

