from toolbox.logger import get_logger

logger = get_logger(__name__)

def rank_features(df, target, method="auto", audit=None)
    # entry point — ranks features by importance

def _random_forest_importance(df, target)
    # uses sklearn RandomForestClassifier/Regressor
    # automatically detects classification vs regression

def _correlation_importance(df, target)
    # uses absolute correlation with target as importance score
    # good for quick linear relationships

def _mutual_information(df, target)
    # uses sklearn mutual_info_regression or mutual_info_classif
    # captures non-linear relationships too