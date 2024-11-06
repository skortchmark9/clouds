import pandas as pd
import json
import netCDF4
from collections import defaultdict

def load_qcloud():
    path = 'data/wrf3d_d01_PGW_QCLOUD_200010.nc'
    return netCDF4.Dataset(path)

def load_qice():
    path = 'data/wrf3d_d01_PGW_QICE_200010.nc'
    return netCDF4.Dataset(path)


def get_cloud_mixing_ratio_flat(ds):
    t = 0
    h = 0
    qcloud = ds.variables['QCLOUD'][t, h, :, :]  # Cloud mixing ratio
    lat = ds.variables['XLAT'][:].data           # Latitude
    lon = ds.variables['XLONG'][:].data          # Longitude
    # time = ds.variables['Time']                  # Time dimension (optional)
    # height_levels = ds.dimensions['bottom_top'].size  # Number of vertical levels

    # Initialize the list to hold the data
    output_data = []

    for i in range(lat.shape[0] - 1):
        for j in range(lat.shape[1] - 1):
            # Get corners of each cell
            corners_of_box = [
                {"lat": float(lat[i, j]), "lon": float(lon[i, j])},
                {"lat": float(lat[i+1, j]), "lon": float(lon[i+1, j])},
                {"lat": float(lat[i, j+1]), "lon": float(lon[i, j+1])},
                {"lat": float(lat[i+1, j+1]), "lon": float(lon[i+1, j+1])}
            ]

            # Get the cloud mixing ratio for the cell
            cloud_mixing_ratio = float(qcloud[i, j])
    
            output_data.append({
                "corners_of_box": corners_of_box,
                "cloud_mixing_ratio": [cloud_mixing_ratio]
            })

    return output_data

def get_cloud_mixing_ratio_3d(ds, state):
    t = 39 # Found this had some cloud_mixing in it (in kansas)
    h = 0
    qcloud = ds.variables['QCLOUD'][t, :, :, :]  # Cloud mixing ratio
    lat = ds.variables['XLAT'][:].data           # Latitude
    lon = ds.variables['XLONG'][:].data          # Longitude
    # time = ds.variables['Time']                  # Time dimension (optional)
    height_levels = ds.dimensions['bottom_top'].size  # Number of vertical levels

    # Initialize the list to hold the data
    output_data = []

    state_bb = get_state_bb(state)

    for i in range(lat.shape[0] - 1):
        for j in range(lat.shape[1] - 1):
            first_corner = {"lat": float(lat[i, j]), "lon": float(lon[i, j])}

            if not is_in_bounds(state_bb, first_corner['lat'], first_corner["lon"]):
                continue

            # Get corners of each cell
            corners_of_box = [
                first_corner,
                {"lat": float(lat[i+1, j]), "lon": float(lon[i+1, j])},
                {"lat": float(lat[i, j+1]), "lon": float(lon[i, j+1])},
                {"lat": float(lat[i+1, j+1]), "lon": float(lon[i+1, j+1])}
            ]

            cloud_mixing_ratios = []
            for h in range(0, height_levels):
                cloud_mixing_ratio = float(qcloud[h, i, j])
                cloud_mixing_ratios.append(cloud_mixing_ratio)
    
            if (sum(cloud_mixing_ratios)):
                output_data.append({
                    "corners_of_box": corners_of_box,
                    "cloud_mixing_ratios": cloud_mixing_ratios
                })

    return output_data


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


def find_rainiest_time(ds, state):
    # time = ds.variables['Time']                  # Time dimension (optional)
    height_levels = ds.dimensions['bottom_top'].size  # Number of vertical levels

    by_time = defaultdict(float)

    state_bb = get_state_bb(state)
    lat_lngs = list(find_lat_lngs_in_bounds(ds, state_bb))

    for t in range(0, 250):
        print(f"Trying time: {t} - last time got: {by_time.get(t - 1)}")
        for h in range(height_levels):
            qcloud_data = ds.variables['QCLOUD'][t, h, :, :].data
            for (i, j) in lat_lngs:
                by_time[t] += qcloud_data[i, j]

    return by_time




def write_json(data, suffix):
    fname = 'cloud_mixing_ratio_' + suffix + '.json'
    with open(fname, 'w+') as f:
        json.dump(data, f)