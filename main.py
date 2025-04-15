import uvicorn
import os
import yaml
from pathlib import Path

def main():
    """Run the FastAPI server."""
    # Load configuration
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure model exists
    model_path = Path(config['model']['path'])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Run FastAPI server
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

if __name__ == "__main__":
    main()
