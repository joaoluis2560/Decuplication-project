import uuid
import flask
from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint
from dedupe import DedupeModel
REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route('/isAlive', methods=['GET'])                                                                                                    
def index():     
    return "ok 200"

@REQUEST_API.route('/request', methods=['POST'])                                                                                                    
def get_prediction():     
    if not request.get_json():
        abort(400)
    data = request.get_json(force=True)                                                                                                            
    #json_data = flask.request.json
    dedupe = DedupeModel()
    pred = dedupe.predict(data)
    return jsonify(pred)



