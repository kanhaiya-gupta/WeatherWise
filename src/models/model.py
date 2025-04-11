from sklearn.ensemble import RandomForestClassifier

def define_model():
    """Define the machine learning model."""
    model = RandomForestClassifier(random_state=42)
    return model
