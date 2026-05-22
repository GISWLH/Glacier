/*
Export ERA5-Land warm-season region-year summaries for the 17 Glacier project regions.

Recommended first use:
1. Set regionKey and exportYears.
2. Run one region at a time.
3. Export CSV to Google Drive.
4. Merge locally into the formal region-year panel.

This first version focuses on monthly warm-season background climate, not daily extremes.
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
var exportYears = ee.List.sequence(2000, 2024);
var driveFolder = 'GlacierClimate';
var exportPrefix = 'era5_land_warm_season';
var scaleMeters = 11132;
var geometryToleranceMeters = 5000;

var profile = regionProfiles[regionKey];
var lakes = ee.FeatureCollection(profile.regionAsset);
var lakePoints = lakes.map(function(ft) {
  return ee.Feature(ft.geometry().centroid(geometryToleranceMeters));
});
var regionDisplayGeom = lakePoints.geometry().bounds(geometryToleranceMeters);
var warmMonths = ee.List(profile.warmMonths);

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');

function kelvinToCelsius(img, bandName, outName) {
  return img.select(bandName).subtract(273.15).rename(outName);
}

function safeNumber(dict, key) {
  dict = ee.Dictionary(dict);
  return ee.Number(ee.Algorithms.If(dict.contains(key), dict.get(key), null));
}

function annualWarmSeasonFeature(year) {
  year = ee.Number(year).toInt();
  var yearCol = era5
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .filter(ee.Filter.inList('month', warmMonths));

  var imageCount = yearCol.size();

  var t2mC = yearCol.map(function(img) {
    return kelvinToCelsius(img, 'temperature_2m', 't2m_c')
      .copyProperties(img, img.propertyNames());
  });

  var precipMeters = yearCol.select('total_precipitation_sum');

  var t2mMean = t2mC.mean();
  var precipSum = precipMeters.sum().multiply(1000).rename('precip_mm');

  var climateImage = t2mMean.addBands(precipSum);
  var samples = climateImage.sampleRegions({
    collection: lakePoints,
    scale: scaleMeters,
    geometries: false,
    tileScale: 4
  });

  return ee.Feature(null, {
    region_code: profile.regionCode,
    region_key: regionKey,
    year: year,
    warm_months: warmMonths.join(','),
    lake_point_count: lakePoints.size(),
    sampled_point_count: samples.size(),
    era5_month_count: imageCount,
    warm_season_t2m_mean_c: ee.Number(samples.aggregate_mean('t2m_c')),
    warm_season_precip_sum_mm: ee.Number(samples.aggregate_mean('precip_mm')),
    era5_dataset: 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
  });
}

var annualFeatures = ee.FeatureCollection(exportYears.map(annualWarmSeasonFeature));

var climatologyBase = annualFeatures.filter(ee.Filter.lte('year', 2023));
var t2mClim = ee.Number(climatologyBase.aggregate_mean('warm_season_t2m_mean_c'));
var precipClim = ee.Number(climatologyBase.aggregate_mean('warm_season_precip_sum_mm'));

var withAnomalies = annualFeatures.map(function(ft) {
  var t2m = ee.Number(ft.get('warm_season_t2m_mean_c'));
  var precip = ee.Number(ft.get('warm_season_precip_sum_mm'));
  return ft.set({
    warm_season_t2m_anomaly_c: t2m.subtract(t2mClim),
    warm_season_precip_anomaly_mm: precip.subtract(precipClim),
    climatology_ref_years: '2000-2023'
  });
});

print('Region display bounds', regionDisplayGeom);
print('Annual warm-season climate table', withAnomalies.limit(5));
Map.centerObject(regionDisplayGeom, 4);
Map.addLayer(regionDisplayGeom, {color: 'red'}, 'region_bounds');

Export.table.toDrive({
  collection: withAnomalies,
  description: exportPrefix + '_' + profile.regionCode + '_' + regionKey + '_2000_2024',
  folder: driveFolder,
  fileNamePrefix: exportPrefix + '_' + profile.regionCode + '_' + regionKey + '_2000_2024',
  fileFormat: 'CSV'
});
