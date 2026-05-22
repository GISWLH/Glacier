/*
Export ERA5-Land daily warm-extreme region-year summaries for the 17 Glacier project regions.

Indicators produced:
1. era5l_tx90p                - percent of warm-season days with Tmax > regional warm-season P90 threshold
2. era5l_wsdi                 - number of warm-season days that belong to runs of >= 6 consecutive hot days
3. warm_extreme_year_flag     - 1 if era5l_tx90p > 10, else 0

Recommended first use:
1. Set regionKey and exportYears.
2. Run one region at a time in GEE Code Editor.
3. Export CSV to Google Drive.
4. Download to local climate interim folder.
5. Merge later into the formal region-year panel.

This first version uses one pooled warm-season P90 threshold per region over 2000-2023.
The daily source is ECMWF/ERA5_LAND/DAILY_AGGR.
To reduce memory pressure, it uses a convex hull of sampled lake centroids for daily region means.
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
var baselineStartYear = 2000;
var baselineEndYear = 2023;
var driveFolder = 'GlacierClimate';
var exportPrefix = 'era5_daily_extremes_warm_season';
var scaleMeters = 11132;
var geometryToleranceMeters = 5000;
var wsdiMinRunLength = 6;
var warmExtremeTx90pThresholdPct = 10;
var maxSamplePoints = 1000;

var profile = regionProfiles[regionKey];
var lakes = ee.FeatureCollection(profile.regionAsset);
var lakePoints = lakes.map(function(ft) {
  return ee.Feature(ft.geometry().centroid(geometryToleranceMeters));
});
var sampledLakePointsCandidate = lakePoints.randomColumn('rand', 42).sort('rand').limit(maxSamplePoints);
var analysisLakePoints = ee.FeatureCollection(
  ee.Algorithms.If(lakePoints.size().gt(maxSamplePoints), sampledLakePointsCandidate, lakePoints)
);
var lakePointCount = lakePoints.size();
var analysisPointCount = analysisLakePoints.size();
var analysisRegionGeom = analysisLakePoints.geometry(geometryToleranceMeters).convexHull(geometryToleranceMeters);
var regionDisplayGeom = analysisRegionGeom.bounds(geometryToleranceMeters);
var warmMonths = ee.List(profile.warmMonths);

var era5Daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');

function kelvinToCelsius(img, bandName, outName) {
  return img.select(bandName).subtract(273.15).rename(outName);
}

function sampleRegionMean(image, bandName) {
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: analysisRegionGeom,
    scale: scaleMeters,
    maxPixels: 1e9,
    tileScale: 4
  });
  return ee.Number(stats.get(bandName));
}

function dailyRegionTmaxCollection(startDate, endDate) {
  return era5Daily
    .filterDate(startDate, endDate)
    .filter(ee.Filter.inList('month', warmMonths))
    .map(function(img) {
      var tmaxC = kelvinToCelsius(img, 'temperature_2m_max', 'tmax_c');
      var regionMean = sampleRegionMean(tmaxC, 'tmax_c');
      return ee.Feature(null, {
        date: img.date().format('YYYY-MM-dd'),
        year: img.date().get('year'),
        month: img.date().get('month'),
        doy: img.date().getRelative('day', 'year').add(1),
        tmax_c: regionMean
      });
    })
    .filter(ee.Filter.notNull(['tmax_c']));
}

var baselineDaily = ee.FeatureCollection(dailyRegionTmaxCollection(
  ee.Date.fromYMD(baselineStartYear, 1, 1),
  ee.Date.fromYMD(baselineEndYear, 12, 31).advance(1, 'day')
));

var tx90ThresholdC = ee.Number(
  ee.Algorithms.If(
    baselineDaily.size().gt(0),
    baselineDaily.reduceColumns({
      reducer: ee.Reducer.percentile([90]),
      selectors: ['tmax_c']
    }).get('p90'),
    null
  )
);

function computeWSDIFromList(hotList) {
  hotList = ee.List(hotList);
  var init = ee.Dictionary({run: 0, total: 0});
  var result = ee.Dictionary(hotList.iterate(function(v, state) {
    state = ee.Dictionary(state);
    var isHot = ee.Number(v);
    var run = ee.Number(state.get('run'));
    var total = ee.Number(state.get('total'));
    var newRun = ee.Number(ee.Algorithms.If(isHot.eq(1), run.add(1), 0));
    var addNow = ee.Number(ee.Algorithms.If(isHot.eq(1).and(newRun.gte(wsdiMinRunLength)), 1, 0));
    return ee.Dictionary({
      run: newRun,
      total: total.add(addNow)
    });
  }, init));
  return ee.Number(result.get('total'));
}

function annualExtremeFeature(year) {
  year = ee.Number(year).toInt();
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31).advance(1, 'day');
  var dailyFc = ee.FeatureCollection(dailyRegionTmaxCollection(start, end)).sort('date');

  var dailyCount = dailyFc.size();
  var hotFc = ee.FeatureCollection(ee.Algorithms.If(
    dailyCount.gt(0),
    dailyFc.map(function(ft) {
      var tmax = ee.Number(ft.get('tmax_c'));
      var hot = ee.Number(tmax.gt(tx90ThresholdC));
      return ft.set('hot_day', hot);
    }),
    ee.FeatureCollection([])
  ));

  var hotDayCount = ee.Number(ee.Algorithms.If(dailyCount.gt(0), hotFc.aggregate_sum('hot_day'), 0));
  var tx90p = ee.Number(ee.Algorithms.If(dailyCount.gt(0), hotDayCount.divide(dailyCount).multiply(100), null));
  var hotList = ee.List(ee.Algorithms.If(dailyCount.gt(0), hotFc.aggregate_array('hot_day'), ee.List([])));
  var wsdi = ee.Number(ee.Algorithms.If(dailyCount.gt(0), computeWSDIFromList(hotList), 0));
  var extremeFlag = ee.Number(
    ee.Algorithms.If(
      dailyCount.gt(0),
      tx90p.gt(warmExtremeTx90pThresholdPct),
      null
    )
  );

  return ee.Feature(null, {
    region_code: profile.regionCode,
    region_key: regionKey,
    year: year,
    warm_months: warmMonths.join(','),
    era5_dataset: 'ECMWF/ERA5_LAND/DAILY_AGGR',
    tx90_ref_years: baselineStartYear + '-' + baselineEndYear,
    tx90_threshold_c: tx90ThresholdC,
    warm_day_count: dailyCount,
    hot_day_count: hotDayCount,
    era5l_tx90p: tx90p,
    era5l_wsdi: wsdi,
    warm_extreme_year_flag: extremeFlag,
    lake_point_count: lakePointCount,
    sampled_point_count: analysisPointCount
  });
}

var annualFeatures = ee.FeatureCollection(exportYears.map(annualExtremeFeature));

print('Region display bounds', regionDisplayGeom);
print('Lake point count', lakePointCount);
print('Sampled point count used for daily extremes', analysisPointCount);
print('Baseline valid day count', baselineDaily.size());
print('TX90 threshold (C)', tx90ThresholdC);
print('Annual warm-extreme table', annualFeatures.limit(5));
Map.centerObject(regionDisplayGeom, 4);
Map.addLayer(regionDisplayGeom, {color: 'orange'}, 'region_bounds');

Export.table.toDrive({
  collection: annualFeatures,
  description: exportPrefix + '_' + profile.regionCode + '_' + regionKey + '_2000_2024',
  folder: driveFolder,
  fileNamePrefix: exportPrefix + '_' + profile.regionCode + '_' + regionKey + '_2000_2024',
  fileFormat: 'CSV'
});
