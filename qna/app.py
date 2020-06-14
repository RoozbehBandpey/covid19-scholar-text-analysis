import os
import sys
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from msrest.authentication import CognitiveServicesCredentials

import datetime, json, os, time

sys.path.append(os.getcwd())
from utils.load_env import append_vars


ENV = "dev"

if ENV == "dev" and os.environ.get('LUIS_AUTHORING_KEY') is None:
	append_vars(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'local.settings.json'))



def create_app():
    # Create a new LUIS app
    app_name = "Contoso {}".format(datetime.datetime.now())
    app_desc = "Flight booking app built with LUIS Python SDK."
    app_version = "0.1"
    app_locale = "en-us"

    app_id = client.apps.add(dict(name=app_name,
                                  initial_version_id=app_version,
                                  description=app_desc,
                                  culture=app_locale))

    print("Created LUIS app {}\n    with ID {}".format(app_name, app_id))
    return app_id, app_version




key_var_name = 'LUIS_AUTHORING_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
authoring_key = os.environ[key_var_name]

endpoint_var_name = 'LUIS_AUTHORING_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
authoring_endpoint = os.environ[endpoint_var_name]


client = LUISAuthoringClient(authoring_endpoint, CognitiveServicesCredentials(authoring_key))
print(f"API Apps -> \n{client.apps}")


