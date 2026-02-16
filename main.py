import logging
import sys

import uvicorn
from fastapi import FastAPI
from apis.invoke_api import router as invoke_router
from config.dev_config import API_HOST, API_PORT

# Configure logging so [sap_upload] and other tool logs appear on console.
# Set LOG_LEVEL=INFO (or DEBUG) in env to see sap_upload_tool logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
# Ensure tools loggers (sap_upload_tool, sap_client, etc.) show INFO
for name in ("tools.sap_upload_tool", "utils.sap_client"):
    logging.getLogger(name).setLevel(logging.INFO)

app = FastAPI(title="Logistics Agent API")

# Include the invoke API router
app.include_router(invoke_router)

if __name__ == "__main__":
    # Start the service using configuration from config.dev_config
    uvicorn.run(app, host=API_HOST, port=API_PORT)


