import azure.functions as func
import logging
from azure.functions import WsgiMiddleware
from app import app
import traceback

wsgi_app = WsgiMiddleware(app)

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    
    try:
        # Try to handle with Flask
        return wsgi_app.handle(req, context)
    except Exception as e:
        # Catch and display the actual error
        error_msg = f"Flask Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_msg)
        return func.HttpResponse(error_msg, status_code=500)