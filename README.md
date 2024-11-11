* entire dataset is 192 TB!
* needing to understand different conventions: half layers, full layers, surface layers
* datasets separated by params
* cloud cells are arranged on a grid - the grid lines are not evenly spaced, but curved
(this is why it's wrong to call them voxels?)

have a 3d scene in deck gl,
so not just a lat/lng combo.
If so, do we regularize the grid? the original thing seems flat. let's ignore it for now,

first we need to get the lat/lng data to the browser
- write out a json list of [
    { corners_of_box, cloud_mixing_ratio },
    ...
    1,999,999
]
can compress it - won't be too big.

then just need to color in the box.

maybe start by just rendering the grid, maybe you can use a sphere mesh geometry. Or even simpler to do 2d?4
OR geojson
Or pointcloud?

it's probably gonna be simplest to start with mesh. although pointcloud would be so cool, it's literally
a cloud

Different time formats are often included to make the data more flexible:

Absolute time (Time) allows for easy alignment with other datasets.
Relative time (XTIME) makes it easier to analyze within-model trends.
Formatted timestamps (Times) are user-friendly and good for display.

// ci system for scientists
- sell to labs - not only academic ones

cat cloud_mixing_ratio_us.json| gzip > cloud_mixing_ratio_us.json.gz

--

https://towardsdatascience.com/geospatial-indexing-with-quadkeys-d933dff01496



total condenstate = qice + qcloud + qgraupel + qrain + qsnow
interpolate across heights, if GPH is > 4km
take pictures of canvas (maybe using playwright?)
create a nn lol


Z is...
    - sliced by time
    - each has 51, which makes sense because they are "half layers"
    - within a given layer, you have a lat long

z['Z'][t][h][lat][long] (meters)

where h is an index

why does height change over time lol

heights are different between levels, on average 400m

Problem:
* heights are not very dramatic
* they 


Do these heights even make sense?
"3,000 to 7,600 meters (10,000 to 25,000 ft) in polar regions, 5,000 to 12,200 meters (16,500 to 40,000 ft) in temperate regions, and 6,100 to 18,300 meters (20,000 to 60,000 ft) in the tropical region"

{
    "corners_of_box": [
        {
            "lat": 36.9936408996582,
            "lon": -100.65997314453125
        },
        {
            "lat": 37.030242919921875,
            "lon": -100.66131591796875
        },
        {
            "lat": 36.99470138549805,
            "lon": -100.6141357421875
        },
        {
            "lat": 37.031314849853516,
            "lon": -100.615478515625
        }
    ],
    "total_condensation": [
        0,
        0,
        0,
        0,
        7.014246072856167e-9,
        3.12464365492815e-8,
        6.588127376971897e-8,
        1.457548108874107e-7,
        3.2049766218733566e-7,
        6.394106435436697e-7,
        9.711566235637292e-7,
        0.00016412534529308687,
        0.0004927363926583439,
        0.00029637607465815563,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "cell_heights": [
        838,
        893,
        959,
        1035,
        1120,
        1216,
        1326,
        1457,
        1607,
        1777,
        1967,
        2182,
        2428,
        2705,
        3012,
        3354,
        3736,
        4158,
        4619,
        5100,
        5581,
        6061,
        6541,
        7020,
        7499,
        7977,
        8454,
        8930,
        9406,
        9880,
        10355,
        10829,
        11306,
        11786,
        12270,
        12759,
        13251,
        13747,
        14246,
        14749,
        15258,
        15775,
        16305,
        16847,
        17403,
        17967,
        18545,
        19147,
        19774,
        20383
    ]
}