# Fine-resolution satellite example

Open palu_8.tif using Open image. This is genuine georeferenced Maxar/xBD satellite RGB from the OpenEarthMap author demo, around 0.5 m sampling. Original native RGB pixels and EPSG:4326 footprint are preserved. There is no measured NIR, so detection uses RGB only; the enhancement pipeline estimates its fourth channel and labels that source contract. GeoTIFF export preserves the geographic CRS, but physical areas are unavailable for geographic coordinates. Acquisition date is not supplied by the source example and has not been invented.

Use this finer imagery to demonstrate road/building predictions. The matching human labels are in tests/fixtures/real_landcover/labels/palu_8.tif. They are upstream examples and may be in the pretrained model training/validation; they are not independent test accuracy. See manifest.json for immutable source URL, SHA-256 and CC BY-NC-SA 4.0 attribution.
