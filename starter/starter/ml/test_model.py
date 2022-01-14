

def test_precision(precision):
    """ The ratio between the True Positives and all the Positives."""
    assert(precision >= .75)

def test_recall(recall):
    """ The measure of our model correctly identifying True Positives."""
    assert(recall >= .60)

def test_beta(beta):
    """The Fbeta-measure measure is an abstraction of the F-measure where the balance of precision and recall
     in the calculation of the harmonic mean is controlled by a coefficient called beta."""
    assert(beta >= .70)