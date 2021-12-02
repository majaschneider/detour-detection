# de4l-detour-detection

<h1>Estimate exposure of private information in trajectory data based on detours.</h1>
<p>This algorithm calculates the difference from the ideal path between
selected geographical points. The higher the difference, the higher the privacy risk.</p>

## Setup

### Nominatim

The package needs [Nominatim](https://github.com/osm-search/Nominatim) to reverse geocode coordinates.

Nominatim is set up to start as a docker container listening at port `8080`. The Nominatim service uses data about Germany.

To start the container, run

```
docker-compose up
```

The first start might take a while, because all files for the Nominatim service have to be generated. All following launches should be quicker, because the container persists its data in `./nominatim-data/` and will only update if new geodata is available.
