from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from flask_caching import Cache
from functools import lru_cache 
import time
import json
import urllib
import os
import pickle

from data_transforms import (
    load_latlon,
    load_data,
    calculate_cloud_condensation_3d
)

app = Flask(__name__)

cors = CORS(app) # allow CORS for all domains on all routes.

cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 60*60  # 1 hour cache timeout
})
cache.init_app(app)

# Define a function to return cache key for incoming requests
def get_cache_key(request):
    return request.url

# Initialize Flask-Compress
compress = Compress()
compress.init_app(app)

# # Set up cache for compressed responses
compress.cache = cache
compress.cache_key = get_cache_key


# Load datasets at startup
latlon = load_latlon()


def disk_cache_ccc(time, state):
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)

    url = f"http://127.0.0.1:5000/cloud_data?time={time}&states={state}"

    path = urllib.parse.quote_plus(url)
    path = cache_dir + '/' + path
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    data = load_data(time)
    output = calculate_cloud_condensation_3d(latlon, data, state)
    with open(path, 'wb') as f:
        pickle.dump(output, f)

    return output


for i in range(0, 100):
    if i % 10 == 0:
        print(i)
    disk_cache_ccc(i, 'california')

@app.route('/cloud_data', methods=['GET'])
def cloud_data():
    time = int(request.args.get('time', 39))
    states = request.args.getlist('states')
    print(states)
    if not states:
        return jsonify({'error': 'Please provide a state name'}), 400
    
    data = sum([
        disk_cache_ccc(time, state)
        for state in states
    ], [])
    
    try:
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/cloud_data_timerange', methods=['GET'])
def cloud_data_timerange():
    states = request.args.getlist('states')
    min_time = int(request.args.get('minTime', 0))
    max_time = int(request.args.get('maxTime', 100))
    if not states:
        return jsonify({'error': 'Please provide a state name'}), 400

    times = range(min_time, max_time)
    output = {}
    for time in times:
        if time % 10 == 0:
            print(time)

        data = sum([disk_cache_ccc(time, state)
            for state in states
        ], [])
        output[time] = data
    
    try:
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
