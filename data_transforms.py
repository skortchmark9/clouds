import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import json
import gzip
import netCDF4
from collections import defaultdict, Counter
import time
import numpy as np

MAX_VALUE = np.float32(0.021897616)

def load_z():
    path = 'data/wrf3d_d01_CTRL_Z_20001001.nc'
    return netCDF4.Dataset(path)

def load_qcloud():
    path = 'data/wrf3d_d01_PGW_QCLOUD_200010.nc'
    return netCDF4.Dataset(path)

def load_qice():
    path = 'data/wrf3d_d01_PGW_QICE_200010.nc'
    return netCDF4.Dataset(path)

def load_latlon():
    cloud = netCDF4.Dataset('data/wrf3d_d01_PGW_QCLOUD_200010.nc')
    height_levels = cloud.dimensions['bottom_top'].size
    output = {
        'XLAT': cloud.variables['XLAT'][:].data,
        'XLONG': cloud.variables['XLONG'][:].data,
        'HEIGHT_LEVELS': height_levels
    }
    return output

def load_data(t = 0):
    # HACK
    if t > 478:
        raise Exception('Go download more data')
    month = 10
    times_in_month = 239
    if t > times_in_month:
        month = 11
        t = t - times_in_month
    # Ice mixing ratio (kg kg-1)
    ice = netCDF4.Dataset(f'data/wrf3d_d01_PGW_QICE_2000{month}.nc')

    # Cloud water mixing ratio (kg kg-1)
    cloud = netCDF4.Dataset(f'data/wrf3d_d01_PGW_QCLOUD_2000{month}.nc')

    # Graupel mixing ratio (kg kg-1)
    graupel = netCDF4.Dataset(f'data/wrf3d_d01_PGW_QGRAUP_2000{month}.nc')

    # Rain water mixing ratio (kg kg-1)
    rain = netCDF4.Dataset(f'data/wrf3d_d01_PGW_QRAIN_2000{month}.nc')

    time_to_match = cloud.variables['Times'][t].tobytes().decode('utf-8')
    print(time_to_match)

    # Snow mixing ratio (kg kg-1) - saved over a file a day vs. a month
    # so we need to find the file which starts at the right time.
    def find_daily_data(variable, path):
        # Snow mixing ratio (kg kg-1) - saved over a file a day vs. a month
        # so we need to find the file which starts at the right time.
        for i in range(1, 31):
            filepath = path + f'_2000{month}{str(i).zfill(2)}.nc'
            ds = netCDF4.Dataset(filepath)
            for j, times in enumerate(ds.variables['Times']):
                str_time = times.tobytes().decode('utf-8')
                if str_time == time_to_match:
                    print(f"Found time match for daily: {path}@{str_time}", )
                    return ds.variables[variable][j, :, :, :]

    snow = find_daily_data('QSNOW', 'data/wrf3d_d01_PGW_QSNOW')
    heights = find_daily_data('Z', 'data/wrf3d_d01_CTRL_Z')


    
    start = time.time()
    output = {
        'QCLOUD': cloud.variables['QCLOUD'][t, :, :, :].data,
        'QICE': ice.variables['QICE'][t, :, :, :].data,
        'QGRAUP': graupel.variables['QGRAUP'][t, :, :, :].data,
        'QRAIN': rain.variables['QRAIN'][t, :, :, :].data,
        'QSNOW': snow.data,
        'HEIGHTS': heights.data,
        'TIME': time_to_match,
    }
    end = time.time()
    print(f"Time to load data for time t={t}: {end - start}s")
    return output


def create_100km_blocks(time_str, latlon, preprocessed_data, n_blocks=100):
    # Max value for normalization, determined by the max value in the dataset
    max_value = MAX_VALUE
    lat = latlon['XLAT']
    lon = latlon['XLONG']
    height_levels = latlon['HEIGHT_LEVELS']

    N = lat.shape[0]
    M = lat.shape[1]
    block_size = 100 // 4 # (100km / 4 = 25km)

    max_iterations = 1000
    iterations = 0
    blocks = []
    while len(blocks) < n_blocks:
        iterations += 1
        if iterations > max_iterations:
            raise Exception("Could not find enough blocks after 1000 iterations")

        # Ensure top-left corner of the subgrid is within bounds
        start_x = np.random.randint(0, N - block_size)
        start_y = np.random.randint(0, M - block_size)

        end_x = start_x + block_size
        end_y = start_y + block_size

        corners = [
            {"lat": lat[start_x, start_y], "lon": lon[start_x, start_y], 'indices': [start_x, start_y]},
            {"lat": lat[end_x, start_y], "lon": lon[end_x, start_y], 'indices': [end_x, start_y]},
            {"lat": lat[start_x, end_y], "lon": lon[start_x, end_y], 'indices': [start_x, end_y]},
            {"lat": lat[end_x, end_y], "lon": lon[end_x, end_y], 'indices': [end_x, end_y]},
        ]

        truth = preprocessed_data[start_x:end_x, start_y:end_y, :]
        truth_normalized = truth / max_value

        top_down = np.sum(truth_normalized, axis=2)

        altitude_profile = [
            np.average(truth_normalized[:, :, h])
            for h in range(height_levels)
        ]
        # No clouds at all in this block!
        if (sum(altitude_profile) == 0):
            continue

        block = {
            # Metadata which won't be fed into the model
            'time': time_str,
            'corners': corners,
            # i, j, h - actual condensation values in each cell
            'truth': truth_normalized,

            # 25 x 25 - total water path (integrated over height)
            'top_down': top_down,

            # For profile within the cell.
            'altitude_profile': altitude_profile
        }
        blocks.append(block)

    return blocks


def create_time_block(latlon, t):
    start_block = time.time()
    data = load_data(t)
    preprocessed_data = sum_condensation(data)
    time_str = data['TIME']
    blocks = create_100km_blocks(time_str, latlon, preprocessed_data)
    write_blocks(time_str, blocks)
    end_block = time.time()
    print(f"Time to create blocks for t={t}: {end_block - start_block}s")

def create_blocks_across_time():
    latlon = load_latlon()
    start = time.time()

    # Define the range of time steps
    time_steps = range(1, 478)

    with ProcessPoolExecutor() as executor:
        # Submit tasks
        futures = {executor.submit(create_time_block, latlon, t): t for t in time_steps}
        
        for future in as_completed(futures):
            t = futures[future]
            try:
                # Retrieve result to trigger any exceptions raised in the worker
                future.result()
                print(f"Time step {t} completed successfully.")
            except Exception as e:
                print(f"Error in time step {t}: {e}")

    end = time.time()
    print(f"Time to create all blocks: {end - start}s")

def sum_condensation(data):
    """ """
    qcloud = data['QCLOUD']
    qice = data['QICE']
    qgraup = data['QGRAUP']
    qrain = data['QRAIN']
    qsnow = data['QSNOW']

    # Vectorized summation in (h, i, j) order
    vectorized_data = qcloud + qice + qgraup + qrain + qsnow

    # Transpose the final output to (i, j, h)
    vectorized_data = vectorized_data.transpose(1, 2, 0)
    return vectorized_data


def load_all_blocks_from_disk():
    blocks = []
    old = False
    n_blocks = 230 if old else 478
    for t in range(1, 478):
        if old:
            fname = f'blocks_old/100_blocks_at_t={t}.npz'
            blocks.extend(np.load(fname, allow_pickle=True)['arr_0'])
        else:
            time_str = create_time_str(t)
            blocks.extend(read_blocks(time_str))
    return blocks

def calculate_cloud_condensation_3d(latlon, data, state = None):
    lat = latlon['XLAT']
    lon = latlon['XLONG']
    height_levels = latlon['HEIGHT_LEVELS']

    qcloud = data['QCLOUD']
    qice = data['QICE']
    qgraup = data['QGRAUP']
    qrain = data['QRAIN']
    qsnow = data['QSNOW']
    heights = data['HEIGHTS']
    time_str = data['TIME']

    # Initialize the list to hold the data
    output_data = []

    if state:
        state_bb = get_state_bb(state)
    else:
        state_bb = None

    for i in range(lat.shape[0] - 1):
        for j in range(lat.shape[1] - 1):
            # Get corners of each cell
            corners_of_box = [
                {"lat": float(lat[i, j]), "lon": float(lon[i, j])},
                {"lat": float(lat[i+1, j]), "lon": float(lon[i+1, j])},
                {"lat": float(lat[i, j+1]), "lon": float(lon[i, j+1])},
                {"lat": float(lat[i+1, j+1]), "lon": float(lon[i+1, j+1])}
            ]

            if state_bb and any(
                not is_in_bounds(state_bb, corner['lat'], corner['lon'])
                for corner in corners_of_box
            ):
                continue

            # HACK we hate canada
            if any(corner['lat'] > 49.014322 for corner in corners_of_box):
                continue

            total_condensation = []
            cell_heights = []
            for h in range(0, height_levels):
                cell_qcloud = float(qcloud[h, i, j])
                cell_qice = float(qice[h, i, j])
                cell_qgraup = float(qgraup[h, i, j])
                cell_qrain = float(qrain[h, i, j])
                cell_qsnow = float(qsnow[h, i, j])

                # Heights are half-levels, so we need to interpolate
                # to get to the bottom of the cell.
                #    h + 2
                #
                #    h + 1  
                #          |c|
                #    h
                cell_height = (heights[h + 1, i, j] - heights[h, i, j]) / 2
                cell_height += heights[h, i, j]
                cell_heights.append(round(float(cell_height)))

                total_condensation_in_cell = sum([
                    cell_qcloud,
                    cell_qice,
                    cell_qgraup,
                    cell_qrain,
                    cell_qsnow,
                ])
                total_condensation.append(total_condensation_in_cell)
    
            if sum(total_condensation):
                output_data.append({
                    "corners_of_box": corners_of_box,
                    "total_condensation": total_condensation,
                    'cell_heights': cell_heights,
                })

    return {
        'time': time_str,
        'boxes': output_data
    }


def get_state_bb(state):
    state_bbs = [{"fips":"01","state":"Alabama","bounds":[[-88.4745951503515,30.222501133601334],[-84.89247974539745,35.008322669916694]]},{"fips":"02","state":"Alaska","bounds":[[-179.13657211802118,51.229087747767466],[179.77488070600702,71.352561]]},{"fips":"60","state":"American Samoa","bounds":[[-170.84530299432993,-14.373864584355843],[-169.42394257312571,-14.157381542325423]]},{"fips":"04","state":"Arizona","bounds":[[-114.8128344705447,31.332406253852533],[-109.04483902389023,37.0039183311733]]},{"fips":"05","state":"Arkansas","bounds":[[-94.61946646626465,33.00413641175411],[-89.65547287402873,36.49965029279292]]},{"fips":"06","state":"California","bounds":[[-124.41060660766607,32.5342307609976],[-114.13445790587905,42.00965914828148]]},{"fips":"08","state":"Colorado","bounds":[[-109.05919619986199,36.99275055519555],[-102.04212644366443,41.00198213121131]]},{"fips":"09","state":"Connecticut","bounds":[[-73.72618613336134,40.98480093739937],[-71.78796737717377,42.050894013430124]]},{"fips":"10","state":"Delaware","bounds":[[-75.7900301793018,38.45143390982909],[-75.05063561675617,39.8388153101431]]},{"fips":"11","state":"District of Columbia","bounds":[[-77.11806895668957,38.79162154730547],[-76.90988990509905,38.99435963428633]]},{"fips":"12","state":"Florida","bounds":[[-87.63470035600356,24.51490854927549],[-80.03257567895679,31.000809213282125]]},{"fips":"13","state":"Georgia","bounds":[[-85.60674924999249,30.35909162440624],[-80.84375612136121,35.000591132701324]]},{"fips":"66","state":"Guam","bounds":[[144.62133533915335,13.245763528025279],[144.9587289744897,13.652098761677614]]},{"fips":"15","state":"Hawaii","bounds":[[-160.24970712717126,18.91727560534605],[-154.80833743387433,22.23238695135951]]},{"fips":"16","state":"Idaho","bounds":[[-117.24278650376503,41.988182656016555],[-111.04407577795777,49.00068691035909]]},{"fips":"17","state":"Illinois","bounds":[[-91.51472716237161,36.97041500324003],[-87.4947178902789,42.508772828518275]]},{"fips":"18","state":"Indiana","bounds":[[-88.09771928109281,37.77191769456694],[-84.78480092560925,41.760531838008376]]},{"fips":"19","state":"Iowa","bounds":[[-96.63306039630396,40.37830479583795],[-90.14002756307562,43.50012771146711]]},{"fips":"20","state":"Kansas","bounds":[[-102.05289432564325,36.99275055519555],[-94.59075211432113,40.0028957110471]]},{"fips":"21","state":"Kentucky","bounds":[[-89.57291911219112,36.49707311372113],[-81.96720514115141,39.14641319952199]]},{"fips":"22","state":"Louisiana","bounds":[[-94.04159013340133,28.929616299252984],[-88.81557807968079,33.01959948618486]]},{"fips":"23","state":"Maine","bounds":[[-71.08446575455754,43.059430090190894],[-66.9819027206272,47.459533825428245]]},{"fips":"24","state":"Maryland","bounds":[[-79.48700299202991,37.91709878227782],[-75.05063561675617,39.72284225191251]]},{"fips":"25","state":"Massachusetts","bounds":[[-73.507239199792,41.23908260581605],[-69.92871308883089,42.88675909238091]]},{"fips":"26","state":"Michigan","bounds":[[-90.416403200532,41.696102361213605],[-82.4158668902689,48.190593622126215]]},{"fips":"27","state":"Minnesota","bounds":[[-97.23965108111081,43.49926865177651],[-89.4903653503535,49.384686592055914]]},{"fips":"28","state":"Mississippi","bounds":[[-91.64394174611746,30.180407208762084],[-88.09771928109281,34.99543677455774]]},{"fips":"29","state":"Missouri","bounds":[[-95.76804054400543,35.99538225441254],[-89.09913230512305,40.613687151061505]]},{"fips":"30","state":"Montana","bounds":[[-116.05114089810897,44.35832834237342],[-104.04136319773197,49.00154597004969]]},{"fips":"31","state":"Nebraska","bounds":[[-104.05213107971079,40.00031853197531],[-95.30861091290913,43.001014031230305]]},{"fips":"32","state":"Nevada","bounds":[[-120.00654287832877,35.00145019239192],[-114.04113626206261,42.0019276110661]]},{"fips":"33","state":"New Hampshire","bounds":[[-72.55607629166292,42.696906900759004],[-70.70400059130591,45.30587118110181]]},{"fips":"34","state":"New Jersey","bounds":[[-75.56031536375363,38.928212038110374],[-73.8948829510295,41.357632843118424]]},{"fips":"35","state":"New Mexico","bounds":[[-109.04842831788318,31.332406253852533],[-103.0004679397794,37.00048209241092]]},{"fips":"36","state":"New York","bounds":[[-79.7633786294863,40.502009391283906],[-71.85616396303963,45.01550900568005]]},{"fips":"37","state":"North Carolina","bounds":[[-84.32178200052,33.85116926668266],[-75.45981513195132,36.5881334409244]]},{"fips":"38","state":"North Dakota","bounds":[[-104.04854178571784,45.934702874618736],[-96.55768522245222,49.00068691035909]]},{"fips":"69","state":"Northern Mariana Islands","bounds":[[145.12024720417202,14.110836636456362],[146.0821779942799,18.81247032309323]]},{"fips":"39","state":"Ohio","bounds":[[-84.82069386553866,38.40504468653686],[-80.52071966199662,41.97787393972939]]},{"fips":"40","state":"Oklahoma","bounds":[[-103.00405723377233,33.61664597114971],[-94.43282317863178,37.002200211792115]]},{"fips":"41","state":"Oregon","bounds":[[-124.55417836738366,41.99161889477894],[-116.46390970729706,46.26801803457034]]},{"fips":"42","state":"Pennsylvania","bounds":[[-80.52071966199662,39.720265072840725],[-74.69529551145511,42.269954234532335]]},{"fips":"72","state":"Puerto Rico","bounds":[[-67.94024421674217,17.91217576734767],[-65.22314866408664,18.51609472983729]]},{"fips":"44","state":"Rhode Island","bounds":[[-71.85975325703257,41.15145851737517],[-71.12035869448694,42.019108804878044]]},{"fips":"45","state":"South Carolina","bounds":[[-83.34908332843328,32.03425802107021],[-78.53942937789378,35.21535605535055]]},{"fips":"46","state":"South Dakota","bounds":[[-104.05930966769667,42.484719157181566],[-96.43564922669226,45.9450115909059]]},{"fips":"47","state":"Tennessee","bounds":[[-90.3087243807438,34.98255087919878],[-81.64775797577975,36.67833470843708]]},{"fips":"48","state":"Texas","bounds":[[-106.64719063660635,25.840437651866516],[-93.5175532104321,36.50050935248352]]},{"fips":"49","state":"Utah","bounds":[[-114.05190414404143,36.99790491333913],[-109.04124972989729,42.0019276110661]]},{"fips":"50","state":"Vermont","bounds":[[-73.43904261392613,42.726973989929895],[-71.49364526975269,45.01550900568005]]},{"fips":"78","state":"Virgin Islands","bounds":[[-65.08316619836198,17.679370591195905],[-64.57707574535745,18.38465859717597]]},{"fips":"51","state":"Virginia","bounds":[[-83.67570908179081,36.540885157941574],[-75.24086819838197,39.46598340442404]]},{"fips":"53","state":"Washington","bounds":[[-124.73364306703067,45.54383071539715],[-116.9161607504075,49.00240502974029]]},{"fips":"54","state":"West Virginia","bounds":[[-82.63840311783117,37.2015020600106],[-77.72107034750347,40.63859988208881]]},{"fips":"55","state":"Wisconsin","bounds":[[-92.88942676166761,42.49159163470634],[-86.82351991359913,47.07725226311263]]},{"fips":"56","state":"Wyoming","bounds":[[-111.05843295392954,40.995109653686534],[-104.05213107971079,45.006059349083486]]}]
    bounds = None
    for entry in state_bbs:
        if state.lower() == entry['state'].lower():
            bounds = entry['bounds']

    if bounds is None:
        raise Exception("Could not find state")

    return bounds

def is_in_bounds(bounds, lat, lng):
    min_lng, min_lat = bounds[0]
    max_lng, max_lat = bounds[1]

    # HACK we hate canada
    if lat > 49.014322:
        return False

    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng

def is_in_state(state, lat, lng):
    bounds = get_state_bb(state)

    return is_in_bounds(bounds, lat, lng)

def find_lat_lngs_in_bounds(ds, bounds):
    lats = ds.variables['XLAT'][:].data           # Latitude
    lons = ds.variables['XLONG'][:].data          # Longitude

    for i in range(lats.shape[0] - 1):
        for j in range(lats.shape[1] - 1):
            lat = lats[i, j]
            lon = lons[i, j]
            if not is_in_bounds(bounds, lat, lon):
                continue

            yield (i, j)


def create_time_str(t):
    base_time = pd.Timestamp('2000-10-01 00:00:00')
    time_offset = pd.Timedelta(hours=t * 3)
    return (base_time + time_offset).strftime('%Y-%m-%d_%H:%M:%S')

def find_rainiest_time(state):
    latlon = load_latlon()
    c = Counter()
    for t in range(0, 235):
        data = load_data(t)
        output = calculate_cloud_condensation_3d(latlon, data, state)
        total = sum([
            sum(cell['total_condensation']) for cell in output
        ])
        c[t] = total

    return c


def write_blocks(time_str, blocks):
    fname = f'blocks/100_blocks@{time_str}.npz'
    np.savez_compressed(fname, blocks)

def read_blocks(time_str):
    fname = f'blocks/100_blocks@{time_str}.npz'
    if os.path.exists(fname):
        return np.load(fname, allow_pickle=True)['arr_0']
    else:
        print(f"Could not find block {fname}")
    return []

def write_json(data, suffix):
    fname = 'cloud_condensation_' + suffix + '.json'
    with open(fname, 'w+') as f:
        json.dump(data, f)

    gzip_fname = fname + '.gz'
    with open(fname, 'rb') as f_in:
        with gzip.open(gzip_fname, 'wb') as f_out:
            f_out.writelines(f_in)