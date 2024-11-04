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



--

https://towardsdatascience.com/geospatial-indexing-with-quadkeys-d933dff01496
