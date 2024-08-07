import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


from gaia.preprocessing import Processing

def prediction(X, y):
    """
    Train a machine learning model to predict y from X.

    Parameters
    ----------
    X : pd.DataFrame
        The input data frame with features.
    y : pd.DataFrame
        The target data frame with labels.

    Returns
    -------
    None
    """
    # 1. Synchronize the data
    X, y = synchronize_data(X, y, 'time', 'time')

    print(X)
    # 2. Normalize the data
    processor = Processing()
    X_scaled = processor.normalize_data(X, scaler_type='standard')
    y_scaled = processor.normalize_data(y, scaler_type='standard')

    print(X_scaled)

    # 3. Apply PCA to the output variables
    pca = PCA(n_components=10)
    y_pca = pca.fit_transform(y_scaled.drop(columns=['time']))

    # 4. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled.drop(columns=['time']), y_pca, test_size=0.2, random_state=42)

    # 5. Train a regression model (example with neural network)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)

    # 6. Make predictions and evaluate the model
    y_pred = model.predict(X_test)

    # Inverse the PCA transformation to compare with original variables
    y_pred_original = pca.inverse_transform(y_pred)
    y_test_original = pca.inverse_transform(y_test)

    # Calculate the error
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    explained_variance = pca.explained_variance_ratio_
    print(f'Explained Variance Ratios: {explained_variance}')
    print(f'Total Variance Explained by First 10 Components: {sum(explained_variance)}')


    # 2. Normalize the data
    processor = Processing()
    X_scaled = processor.normalize_data(X, scaler_type='standard')
    y_scaled = processor.normalize_data(y, scaler_type='standard')

    # 3. Apply PCA to the output variables
    pca = PCA(n_components=10)
    y_pca = pca.fit_transform(y_scaled.drop(columns=['time']))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled.drop(columns=['time']), y_pca, test_size=0.2, random_state=42)
    best_model = tune_model(X_train, y_train)

    # Train the best model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    y_pred_original = pca.inverse_transform(y_pred)
    y_test_original = pca.inverse_transform(y_test)
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    # Evaluate the best model
    evaluate_model(best_model, X_train, y_train, pca)

def synchronize_data(X, y, time_col_X, time_col_y):
    """
    Synchronizes the data X and y based on the time column.

    Parameters
    ----------
    X : pd.DataFrame
        The input data frame with time column.
    y : pd.DataFrame
        The target data frame with time column.
    time_col_X : str
        The name of the time column in X.
    time_col_y : str
        The name of the time column in y.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The synchronized data frames X and y.
    """
    
    # Extract time columns
    time_X = X[time_col_X]
    time_y = y[time_col_y]
    
    # Find common times
    common_times = pd.Series(list(set(time_X).intersection(set(time_y))))
    
    # Filter X and y to keep only samples with common times
    X_sync = X[X[time_col_X].isin(common_times)]
    y_sync = y[y[time_col_y].isin(common_times)]
    
    # Sort synchronized data by time (optional)
    X_sync = X_sync.sort_values(by=time_col_X).reset_index(drop=True)
    y_sync = y_sync.sort_values(by=time_col_y).reset_index(drop=True)
    
    return X_sync, y_sync

def tune_model(X, y):
    """
    Tune the hyperparameters of a neural network using GridSearchCV.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    y : np.ndarray
        The target data.

    Returns
    -------
    GridSearchCV
        The fitted GridSearchCV object.
    """
    param_grid = {
        'hidden_layer_sizes': [(64, 64), (100, 100), (128, 128)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }

    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search

def evaluate_model(model, X, y, pca):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Fit the model
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X)
    
    # Inverse transform the predictions
    y_pred_original = pca.inverse_transform(y_pred)
    y_original = pca.inverse_transform(y)
    
    # Calculate errors
    mse = mean_squared_error(y_original, y_pred_original)
    mae = mean_absolute_error(y_original, y_pred_original)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Cross-Validation MSE: {cv_mse:.4f} ± {cv_std:.4f}')
    
    # Plot residuals
    residuals = y_original - y_pred_original
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, 'o')
    plt.title('Residuals')
    plt.xlabel('Sample')
    plt.ylabel('Error')
    plt.savefig('residuals.png')
