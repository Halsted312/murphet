# tests/test_basics.py
import pytest
from murphet.churn_model import ChurnProphetModel
import numpy as np

def test_package_imports():
    """Basic test to verify package imports work"""
    assert ChurnProphetModel is not None

def test_dummy():
    """Placeholder test to ensure pytest passes"""
    assert True
