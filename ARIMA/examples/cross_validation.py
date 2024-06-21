
from pyro.ops.stats import energy_score_empirical


def cross_validation_folds(data, test_ratio_or_num, num_folds, train_ratio=1.0):
    '''
    Generate time series cross validation folds with a contigous range of test samples
    allocated at various positions of the available data, such that the remaining samples
    will be used as training samples.

    Indexwise the test samples are not always allocated after the train samples, and
    therefore the model should be able to handle such allocation. This enables creating
    more independent cross validation folds from the available data, which is important
    in cases of small data sets.

    Args:
        data: Available data.
        test_ratio_or_num: Ratio (if less than one) or number of samples to be used as test samples.
        num_folds: Number of folds to create.
        train_ratio: Ratio of samples to be used for training (defaults to one).
    
    Returns:
        fold_data: Fold data.
        train_idx: Indices of the train samples.
        test_idx: Indices of the test samples.
        start_idx: Fold starting index in the input data.

    Examples:
        >>> from torch import randn
        >>> data = randn(10)
        >>> for fold_data, train_idx, test_idx, start_idx in cross_validation_folds(data, 0.2, 3):
        ...     print(train_idx, test_idx)
        [2, 3, 4, 5, 6, 7, 8, 9] [0, 1]
        [0, 1, 2, 3, 6, 7, 8, 9] [4, 5]
        [0, 1, 2, 3, 4, 5, 6, 7] [8, 9]
    '''
    num_samples = len(data)
    num_test_samples = round(num_samples * test_ratio_or_num) if test_ratio_or_num < 1 else test_ratio_or_num
    num_train_samples = round((num_samples - num_test_samples) * train_ratio)
    num_fold_samples = num_train_samples + num_test_samples
    for fold_num in range(num_folds):
        start_idx = round(fold_num / (num_folds - 1) * (num_samples - num_fold_samples))
        all_idx = [*range(start_idx, start_idx + num_fold_samples)]
        test_start_idx = start_idx + round(fold_num / (num_folds - 1) * num_train_samples)
        test_idx = [*range(test_start_idx, test_start_idx + num_test_samples)]
        train_idx = [idx for idx in all_idx if idx not in test_idx]
        # Rebase to starting index of zero
        train_idx = [idx - start_idx for idx in train_idx]
        test_idx = [idx - start_idx for idx in test_idx]
        yield data[all_idx], train_idx, test_idx, start_idx


def score_fold(posterior_predictive_sampler, obs, train_idx, test_idx, score_func=energy_score_empirical):
    '''
    Score cross-validation fold using the posterior predictive distribution.
    '''
    predictions = posterior_predictive_sampler(obs, train_idx, test_idx)
    score = score_func(predictions, obs[test_idx])
    return score, predictions


if __name__ == "__main__":
    import doctest
    doctest.testmod()
