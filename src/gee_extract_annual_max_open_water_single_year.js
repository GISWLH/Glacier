/*
Single-year export version of the annual maximum open-water extraction script.

Use this when:
1. The full regional CSV is too large or too slow to download.
2. You want faster QC on a subset of years.
3. You want safer, resumable exports.

Recommended strategy:
- Start with one region and one year.
- Then export 3-year chunks or 1-year files.
- Merge locally after download.
*/

var regionProfiles = {
  western_canada_us: {
    months: [7, 8, 9],
    exportPrefix: 'annual_area_02_western_canada_us',
    lakeAsset: 'projects/ee-winsnowter/assets/02_western_canada_us_gee_upload'
  },
  greenland_periphery: {
    months: [7, 8, 9],
    exportPrefix: 'annual_area_05_greenland_periphery',
    lakeAsset: 'projects/ee-winsnowter/assets/05_greenland_periphery_gee_upload'
  },
  central_asia: {
    months: [7, 8, 9],
    exportPrefix: 'annual_area_13_central_asia',
    lakeAsset: 'projects/ee-winsnowter/assets/13_central_asia_gee_upload'
  }
};

var regionKey = 'greenland_periphery';
var exportYear = 2000;
var profile = regionProfiles[regionKey];

var lakeAsset = profile.lakeAsset;
var exportPrefix = profile.exportPrefix + '_' + exportYear;
var months = profile.months;
var roiBufferMeters = 120;
var scaleMeters = 30;
var minimumImagesPerYear = 3;
var minimumValidAreaFraction = 0.70;

var lakes = ee.FeatureCollection(lakeAsset);
var pixelAreaKm2 = ee.Image.pixelArea().divide(1e6).rename('area_km2');

function propOr(feature, primary, fallback) {
  return ee.Algorithms.If(feature.propertyNames().contains(primary), feature.get(primary), feature.get(fallback));
}

function applyScaleFactors(img) {
  var optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2);
  return img.addBands(optical, null, true);
}

function maskLandsatL2(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 1).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));
  var sat = img.select('QA_RADSAT').eq(0);
  return img.updateMask(mask).updateMask(sat);
}

function renameBandsAndTagSensor(img) {
  var spacecraft = ee.String(img.get('SPACECRAFT_ID'));
  var bands = img.bandNames();
  var isL57 = bands.contains('SR_B1');
  var renamed = ee.Image(ee.Algorithms.If(
    isL57,
    img.select(['SR_B2', 'SR_B4', 'SR_B5', 'SR_B7'], ['green', 'nir', 'swir1', 'swir2']),
    img.select(['SR_B3', 'SR_B5', 'SR_B6', 'SR_B7'], ['green', 'nir', 'swir1', 'swir2'])
  )).copyProperties(img, img.propertyNames());

  var sensorCode = ee.Number(
    ee.Algorithms.If(spacecraft.compareTo('LANDSAT_5').eq(0), 5,
      ee.Algorithms.If(spacecraft.compareTo('LANDSAT_7').eq(0), 7,
        ee.Algorithms.If(spacecraft.compareTo('LANDSAT_8').eq(0), 8,
          ee.Algorithms.If(spacecraft.compareTo('LANDSAT_9').eq(0), 9, -1)
        )
      )
    )
  );

  return renamed.set('sensor_code', sensorCode);
}

function addWaterAndValidity(img) {
  var ndwi = img.normalizedDifference(['green', 'nir']).rename('NDWI');
  var mndwi = img.normalizedDifference(['green', 'swir1']).rename('MNDWI');
  var valid = img.select('green').mask().rename('valid');
  var water = ndwi.gte(0.0).and(mndwi.gte(0.0)).rename('water');
  return img.addBands([ndwi, mndwi, valid, water]);
}

function landsatCollection(year, monthsList, geom) {
  var start = ee.Date.fromYMD(year, monthsList[0], 1);
  var endMonth = monthsList[monthsList.length - 1];
  var end = ee.Date.fromYMD(year, endMonth, 1).advance(1, 'month');

  function prep(collectionId) {
    return ee.ImageCollection(collectionId)
      .filterDate(start, end)
      .filterBounds(geom)
      .map(applyScaleFactors)
      .map(maskLandsatL2)
      .map(renameBandsAndTagSensor)
      .map(addWaterAndValidity);
  }

  return prep('LANDSAT/LT05/C02/T1_L2')
    .merge(prep('LANDSAT/LE07/C02/T1_L2'))
    .merge(prep('LANDSAT/LC08/C02/T1_L2'))
    .merge(prep('LANDSAT/LC09/C02/T1_L2'));
}

function safeNumber(dict, key) {
  return ee.Number(ee.Algorithms.If(ee.Dictionary(dict).contains(key), ee.Dictionary(dict).get(key), 0));
}

function safeFirstNumber(dict, keys) {
  dict = ee.Dictionary(dict);
  keys = ee.List(keys);
  var found = keys.map(function(k) {
    k = ee.String(k);
    return ee.Algorithms.If(dict.contains(k), dict.get(k), null);
  }).removeAll([null]);
  return ee.Number(ee.Algorithms.If(ee.List(found).size().gt(0), ee.List(found).get(0), 0));
}

function medianFromList(values) {
  values = ee.List(values).sort();
  var n = values.size();
  return ee.Number(ee.Algorithms.If(
    n.eq(0),
    0,
    ee.Algorithms.If(
      n.mod(2).eq(1),
      ee.Number(values.get(n.divide(2).floor())),
      ee.Number(values.get(n.divide(2).subtract(1)))
        .add(ee.Number(values.get(n.divide(2))))
        .divide(2)
    )
  ));
}

function summarizeLakeYear(lake, year, monthsList) {
  var lakeGeom = lake.geometry();
  var roi = lakeGeom.buffer(roiBufferMeters);
  var baselineArea = ee.Number(propOr(lake, 'area_0_km2', 'area0_km2'));
  var col = landsatCollection(year, monthsList, roi);
  var imageCount = col.size();

  var emptyStats = ee.Dictionary({
    annual_max_area_km2: 0,
    annual_max_pixel_count: 0,
    valid_area_any_km2: 0,
    valid_pixel_any_count: 0,
    water_area_median_km2: 0,
    l5_count: 0,
    l7_count: 0,
    l8_count: 0,
    l9_count: 0,
    image_count: 0
  });

  var stats = ee.Dictionary(ee.Algorithms.If(imageCount.gt(0), (function() {
    var validAny = col.select('valid').max().rename('valid_any');
    var waterMax = col.select('water').max().rename('water_max');
    var waterAreaCol = col.map(function(img) {
      var areaStats = pixelAreaKm2.updateMask(img.select('water')).reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: lakeGeom,
        scale: scaleMeters,
        maxPixels: 1e9,
        tileScale: 4
      });
      return ee.Feature(null, { area_km2: safeFirstNumber(areaStats, ['area_km2', 'area', 'sum']) });
    });

    var waterAreaMedian = ee.Number(ee.Algorithms.If(
      waterAreaCol.size().gt(0),
      medianFromList(waterAreaCol.aggregate_array('area_km2')),
      0
    ));

    var maxStats = pixelAreaKm2.updateMask(waterMax).reduceRegion({
      reducer: ee.Reducer.sum().combine({
        reducer2: ee.Reducer.count(),
        sharedInputs: true
      }),
      geometry: lakeGeom,
      scale: scaleMeters,
      maxPixels: 1e9,
      tileScale: 4
    });

    var validStats = pixelAreaKm2.updateMask(validAny).reduceRegion({
      reducer: ee.Reducer.sum().combine({
        reducer2: ee.Reducer.count(),
        sharedInputs: true
      }),
      geometry: lakeGeom,
      scale: scaleMeters,
      maxPixels: 1e9,
      tileScale: 4
    });

    var sensorCodes = ee.List(col.aggregate_array('sensor_code'));
    var sensorHist = ee.Dictionary(sensorCodes.reduce(ee.Reducer.frequencyHistogram()));

    return ee.Dictionary({
      annual_max_area_km2: safeFirstNumber(maxStats, ['area_km2_sum', 'area_km2', 'area', 'sum']),
      annual_max_pixel_count: safeFirstNumber(maxStats, ['area_km2_count', 'count']),
      valid_area_any_km2: safeFirstNumber(validStats, ['area_km2_sum', 'area_km2', 'area', 'sum']),
      valid_pixel_any_count: safeFirstNumber(validStats, ['area_km2_count', 'count']),
      water_area_median_km2: waterAreaMedian,
      l5_count: safeNumber(sensorHist, '5'),
      l7_count: safeNumber(sensorHist, '7'),
      l8_count: safeNumber(sensorHist, '8'),
      l9_count: safeNumber(sensorHist, '9'),
      image_count: imageCount
    });
  })(), emptyStats));

  var annualMaxArea = safeNumber(stats, 'annual_max_area_km2');
  var validAreaAny = safeNumber(stats, 'valid_area_any_km2');
  var imageCountNum = safeNumber(stats, 'image_count');
  var validAreaFraction = ee.Number(ee.Algorithms.If(baselineArea.gt(0), validAreaAny.divide(baselineArea), 0));
  var annualAreaFraction = ee.Number(ee.Algorithms.If(baselineArea.gt(0), annualMaxArea.divide(baselineArea), 0));
  var qcEnoughImages = imageCountNum.gte(minimumImagesPerYear);
  var qcEnoughCoverage = validAreaFraction.gte(minimumValidAreaFraction);
  var qcUsable = qcEnoughImages.and(qcEnoughCoverage);

  return ee.Feature(null, {
    lake_id: propOr(lake, 'lake_id', 'lake_id'),
    lake_type: propOr(lake, 'lake_type', 'lake_type'),
    harmonized_class: propOr(lake, 'harmonized_class', 'hclass'),
    rgi_region_name: propOr(lake, 'rgi_region_name', 'rgi_name'),
    glambie_region_key: propOr(lake, 'glambie_region_key', 'glmb_key'),
    baseline_area_0_km2: baselineArea,
    elevation_m: propOr(lake, 'elevation_m', 'elev_m'),
    latitude: propOr(lake, 'latitude', 'lat'),
    longitude: propOr(lake, 'longitude', 'lon'),
    year: year,
    months: monthsList.join(','),
    image_count: imageCountNum,
    annual_max_area_km2: annualMaxArea,
    annual_max_pixel_count: safeNumber(stats, 'annual_max_pixel_count'),
    valid_area_any_km2: validAreaAny,
    valid_pixel_any_count: safeNumber(stats, 'valid_pixel_any_count'),
    water_area_median_km2: safeNumber(stats, 'water_area_median_km2'),
    baseline_valid_area_fraction: validAreaFraction,
    annual_area_to_baseline_ratio: annualAreaFraction,
    l5_count: safeNumber(stats, 'l5_count'),
    l7_count: safeNumber(stats, 'l7_count'),
    l8_count: safeNumber(stats, 'l8_count'),
    l9_count: safeNumber(stats, 'l9_count'),
    qc_enough_images: qcEnoughImages,
    qc_enough_coverage: qcEnoughCoverage,
    qc_usable: qcUsable
  });
}

var rows = lakes.map(function(lake) {
  return summarizeLakeYear(lake, exportYear, months);
});

Export.table.toDrive({
  collection: rows,
  description: exportPrefix,
  folder: 'GlacierAnnualArea',
  fileNamePrefix: exportPrefix,
  fileFormat: 'CSV'
});
