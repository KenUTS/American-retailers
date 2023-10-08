from sklearn.metrics import mean_squared_error
def print_MSE_scores(y_preds, y_actuals, set_name=None):
    """Print the AUROC for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """


    print(f"MSE {set_name}: {mean_squared_error(y_actuals, y_preds)}")

def assess_MSE_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its AUROC scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict_proba(features)[:,1]
    print_MSE_scores(y_preds=preds, y_actuals=target, set_name=set_name)