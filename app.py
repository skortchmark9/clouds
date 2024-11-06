from flask import Flask, jsonify, request
from flask_compress import Compress
from flask_cors import CORS
from flask_caching import Cache

from data_transforms import (
    load_qcloud,
    load_qice,
    get_cloud_mixing_ratio_3d
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

# Set up cache for compressed responses
compress.cache = cache
compress.cache_key = get_cache_key


# Load datasets at startup
ds_latlon = load_qcloud()  # Assuming this has latitude and longitude variables
ds_qcloud = load_qcloud()
ds_qice = load_qice()

@app.route('/', methods=['GET'])
def hey():
    return 'hi'


@app.route('/cloud_data', methods=['GET'])
def cloud_data():
    time = int(request.args.get('time', 39))
    states = request.args.getlist('states')
    print(states)
    if not states:
        return jsonify({'error': 'Please provide a state name'}), 400
    
    data = sum([
        get_cloud_mixing_ratio_3d(ds_latlon, ds_qcloud, ds_qice, state, time)
        for state in states
    ], [])
    
    try:
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
