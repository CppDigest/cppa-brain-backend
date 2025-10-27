import os
import sys
import uvicorn

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the API app directly
from api import app

if __name__ == "__main__":
    # Get configuration
    from config import get_config
    
    host = get_config("api.host", "0.0.0.0")
    port = get_config("api.port", 8080)
    debug = get_config("api.debug", False)
    workers = get_config("api.workers", 1)
    
    # Run the API
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        workers=workers if not debug else 1
    )