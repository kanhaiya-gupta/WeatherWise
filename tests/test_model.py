from src.models.model import define_model

def test_model_definition():
    """Test model definition."""
    model = define_model()
    assert model is not None, "Model should be defined"
    assert hasattr(model, 'fit'), "Model should have a fit method"
