import pandas as pd


def get_cloud_mixing_ratio(ds):
    t = 0
    h = 0
    qcloud = ds.variables['QCLOUD'][t, h, :, :]  # Cloud mixing ratio
    lat = ds.variables['XLAT'][:].data           # Latitude
    lon = ds.variables['XLONG'][:].data          # Longitude
    # time = ds.variables['Time']                  # Time dimension (optional)
    # height_levels = ds.dimensions['bottom_top'].size  # Number of vertical levels

    # Prepare data sample
    sample_data = {
        "Latitude": [],
        "Longitude": [],
        "Cloud Mixing Ratio": []
    }

    # Sample the first few points
    sample_points = 5  # Adjust this to see more points if needed

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
                "cloud_mixing_ratio": cloud_mixing_ratio
            })

    return output_data
    for y in range(sample_points):
        for x in range(sample_points):
            sample_data["Latitude"].append(lat[y, x].data)
            sample_data["Longitude"].append(lon[y, x].data)
            sample_data["Cloud Mixing Ratio"].append(qcloud[t, h, y, x].data)

    # Convert to DataFrame for easy viewing
    df = pd.DataFrame(sample_data)

    return df