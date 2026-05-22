/*
Diagnose which GEE daily ERA5 dataset is suitable for Glacier extreme-warm indicators.

This script compares:
1. ECMWF/ERA5/DAILY
2. ECMWF/ERA5_LAND/DAILY_AGGR

It checks, for one region:
- warm-season day counts by year
- first image timestamps
- available band names
- whether daily Tmax-like bands exist and remain populated through recent years

Run one region at a time in GEE Code Editor.
*/

var regionProfiles = {
  alaska: {
    regionCode: '01',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/01_alaska_gee_upload'
  },
  western_canada_us: {
    regionCode: '02',
    warmMonths: [7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/02_western_canada_us_gee_upload'
  },
  arctic_canada_north: {
    regionCode: '03',
    warmMonths: [7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/03_arctic_canada_north_gee_upload'
  },
  arctic_canada_south: {
    regionCode: '04',
    warmMonths: [7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/04_arctic_canada_south_gee_upload'
  },
  greenland_periphery: {
    regionCode: '05',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/05_greenland_periphery_gee_upload'
  },
  iceland: {
    regionCode: '06',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/06_iceland_gee_upload'
  },
  scandinavia: {
    regionCode: '08',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/08_scandinavia_gee_upload'
  },
  russian_arctic: {
    regionCode: '09',
    warmMonths: [7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/09_russian_arctic_gee_upload'
  },
  north_asia: {
    regionCode: '10',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/10_north_asia_gee_upload'
  },
  central_europe: {
    regionCode: '11',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/11_central_europe_gee_upload'
  },
  caucasus_middle_east: {
    regionCode: '12',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/12_caucasus_middle_east_gee_upload'
  },
  central_asia: {
    regionCode: '13',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/13_central_asia_gee_upload'
  },
  south_asia_west: {
    regionCode: '14',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/14_south_asia_west_gee_upload'
  },
  south_asia_east: {
    regionCode: '15',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/15_south_asia_east_gee_upload'
  },
  low_latitudes: {
    regionCode: '16',
    warmMonths: [6, 7, 8, 9],
    regionAsset: 'projects/ee-winsnowter/assets/16_low_latitudes_gee_upload'
  },
  southern_andes: {
    regionCode: '17',
    warmMonths: [1, 2, 3, 4],
    regionAsset: 'projects/ee-winsnowter/assets/17_southern_andes_gee_upload'
  },
  new_zealand: {
    regionCode: '18',
    warmMonths: [1, 2, 3, 4],
    regionAsset: 'projects/ee-winsnowter/assets/18_new_zealand_gee_upload'
  }
};

var regionKey = 'central_asia';
var years = ee.List.sequence(2019, 2024);
var scaleMeters = 11132;
var geometryToleranceMeters = 5000;
var maxSamplePoints = 300;

var profile = regionProfiles[regionKey];
var lakes = ee.FeatureCollection(profile.regionAsset);
var lakePoints = lakes.map(function(ft) {
  return ee.Feature(ft.geometry().centroid(geometryToleranceMeters));
});
var sampledLakePointsCandidate = lakePoints.randomColumn('rand', 42).sort('rand').limit(maxSamplePoints);
var analysisLakePoints = ee.FeatureCollection(
  ee.Algorithms.If(lakePoints.size().gt(maxSamplePoints), sampledLakePointsCandidate, lakePoints)
);
var analysisRegionGeom = analysisLakePoints.geometry(geometryToleranceMeters).convexHull(geometryToleranceMeters);
var representativePoint = ee.Feature(analysisLakePoints.first()).geometry();
var warmMonths = ee.List(profile.warmMonths);

var DATASETS = [
  {
    key: 'ERA5_DAILY',
    collectionId: 'ECMWF/ERA5/DAILY',
    tmaxBand: 'maximum_2m_air_temperature'
  },
  {
    key: 'ERA5_LAND_DAILY_AGGR',
    collectionId: 'ECMWF/ERA5_LAND/DAILY_AGGR',
    tmaxBand: 'temperature_2m_max'
  }
];

function kelvinToCelsius(img, bandName, outName) {
  return img.select(bandName).subtract(273.15).rename(outName);
}

function pointValueFromImage(image, bandName) {
  var stats = image.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: representativePoint,
    scale: scaleMeters,
    maxPixels: 1e9,
    tileScale: 4
  });
  return stats.get(bandName);
}

function buildCoverageFeature(datasetInfo, year) {
  year = ee.Number(year).toInt();
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31).advance(1, 'day');

  var col = ee.ImageCollection(datasetInfo.collectionId)
    .filterDate(start, end)
    .filter(ee.Filter.inList('month', warmMonths));

  var count = col.size();
  var first = ee.Image(col.first());
  var last = ee.Image(col.sort('system:time_start', false).first());
  var hasAny = count.gt(0);

  var annualMeanImage = ee.Image(ee.Algorithms.If(
    hasAny,
    col.map(function(img) {
      return kelvinToCelsius(img, datasetInfo.tmaxBand, 'tmax_c');
    }).mean(),
    ee.Image.constant(0).rename('tmax_c').updateMask(ee.Image.constant(0))
  ));
  var annualMeanAtPoint = pointValueFromImage(annualMeanImage, 'tmax_c');
  var annualMeanExists = ee.Number(ee.Algorithms.If(annualMeanAtPoint, 1, 0));

  return ee.Feature(null, {
    dataset_key: datasetInfo.key,
    collection_id: datasetInfo.collectionId,
    tmax_band: datasetInfo.tmaxBand,
    region_key: regionKey,
    year: year,
    warm_months: warmMonths.join(','),
    image_count: count,
    annual_mean_exists_flag: annualMeanExists,
    annual_mean_tmax_c_at_point: annualMeanAtPoint,
    first_image_time: ee.Algorithms.If(hasAny, first.date().format('YYYY-MM-dd'), null),
    last_image_time: ee.Algorithms.If(hasAny, last.date().format('YYYY-MM-dd'), null),
    band_names: ee.Algorithms.If(hasAny, first.bandNames().join(','), null)
  });
}

DATASETS.forEach(function(ds) {
  var coverage = ee.FeatureCollection(years.map(function(y) {
    return buildCoverageFeature(ds, y);
  }));
  print('Coverage check: ' + ds.key, coverage);
});

Map.centerObject(analysisRegionGeom, 4);
Map.addLayer(analysisRegionGeom, {color: 'cyan'}, 'analysisRegionGeom');
