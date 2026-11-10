"""NDVI from Sentinel-2 via Google Earth Engine. Needs: pip install earthengine-api, and a
Cloud project registered for Earth Engine (GEE_PROJECT env var). NDVI = (B8 - B4) / (B8 + B4)."""
import os, datetime

def mean_ndvi(lat, lon, days=30, buffer_m=500):
    import ee
    ee.Initialize(project=os.environ["GEE_PROJECT"])
    end = datetime.date.today(); start = end - datetime.timedelta(days=days)
    pt = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt)
           .filterDate(str(start), str(end)).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))
    if col.size().getInfo() == 0:
        return None
    ndvi = col.median().normalizedDifference(["B8", "B4"])
    v = ndvi.reduceRegion(ee.Reducer.mean(), pt, 10).get("nd").getInfo()
    return None if v is None else round(v, 3)
