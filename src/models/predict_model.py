def print_AUROC_scores(y_preds, y_actuals, set_name=None):
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
    from sklearn.metrics import roc_auc_score

    print(f"AUROC {set_name}: {roc_auc_score(y_actuals, y_preds)}")

def assess_AUROC_set(model, features, target, set_name=''):
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
    print_AUROC_scores(y_preds=preds, y_actuals=target, set_name=set_name)