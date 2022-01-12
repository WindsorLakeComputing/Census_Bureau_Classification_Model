import pandas as pd
import pytest

def test_precision(precision):
    assert(precision >= .75)

def test_recall(recall):
    assert(recall >= .60)

def test_beta(beta):
    assert(beta >= .70)