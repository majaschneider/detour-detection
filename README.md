# de4l-detour-privacy-metric

<h1>Privacy metric algorithm based on detours in trajetory data.</h1>
<p>This algorithm calculates the difference from the ideal path between
selected geographical points. The higher the difference, the higher privacy risk.</p>

## Setup

### Nominatim

The metric needs [Nominatim](https://github.com/osm-search/Nominatim) to reverse geocode coordinates.

Nominatim is set up to start as a docker container listening at port `8080`. The Nominatim service uses data about Germany.

To start the container, run

```
docker-compose up
```

The first start might take a while, because all files for the Nominatim service have to be generated. All following launches should be quicker, because the container persists its data in `./nominatim-data/` and will only update data on.
