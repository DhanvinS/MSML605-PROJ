"""Lambda deployment package entry point.

This file is a thin wrapper that imports the actual handler from src/.
Package this with: zip -r lambda.zip retrain_handler.py src/ configs/ requirements.txt
"""

# Re-export the handlers so AWS Lambda can find them as:
#   Handler: retrain_handler.lambda_handler
#   Handler: retrain_handler.update_endpoint_handler

from src.retraining.retrain_trigger import lambda_handler, update_endpoint_handler

__all__ = ["lambda_handler", "update_endpoint_handler"]
