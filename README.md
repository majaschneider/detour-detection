# detour-detection

<h1>Estimate exposure of private information in trajectory data based on detours.</h1>
<p>This algorithm calculates the difference from the ideal path between
selected geographical points. The higher the difference, the higher the privacy risk.</p>

# Installation

Install this package in your repository via pip:
```bash
pip install git+https://github.com/majaschneider/detour-detection.git
```

## Setup

Local instances of Nominatim and Openrouteservices will be created. Further details can be found below.

All services can be started locally with

```
docker-compose up
```

Stop the services with

```
docker-compose down
```

### Nominatim

The package needs [Nominatim](https://github.com/osm-search/Nominatim) to reverse geocode coordinates.

Nominatim is set up to start as a docker container listening at port `8080`. The Nominatim service uses data about Germany.

The first start might take a while, because all files for the Nominatim service have to be generated. All following launches should be quicker, because the container persists its data in `./nominatim-data/` and will only update if new geodata is available.

### Openrouteservice (ORS)

[ORS](https://github.com/GIScience/openrouteservice) is needed to get directions between consecutive geographical points.
It listens to port `8008` of the host environment.

The ORS service is configured to calculate data for Germany. Therefore, a recent pbf-file for Germany has to be [downloaded](https://download.geofabrik.de/europe/germany-latest.osm.pbf).
The pbf then must be placed in the subdirectory `./ors-data/`

For the first build with a pbf-file, ORS has to build its graph. This might take some hours.
For every following build, `BUILD_GRAPHS` in `docker-compose.yml` can be set to false until a new pbf-file is used.
