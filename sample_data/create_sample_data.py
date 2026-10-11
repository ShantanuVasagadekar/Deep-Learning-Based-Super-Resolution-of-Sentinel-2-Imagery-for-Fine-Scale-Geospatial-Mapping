# sample_data/create_sample_data.py
# Generates synthetic four-band illustration (not observed satellite data) for Poland AOI
# Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
# Conforms to Slide 4 ("Sentinel-2 MSI Level-2A imagery at 10 m resolution (B2, B3, B4, B8) in Poland")

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def generate_poland_aoi_bands(output_dir: str = "sample_data/poland_aoi", size: int = 256):
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    h, w = size, size
    
    # Grid coordinates
    y, x = np.ogrid[:h, :w]
    
    # 1. Base Landscape: Agricultural Mosaic & Bare Soil
    # Patchwork field pattern
    field_grid_y = (y // 32)
    field_grid_x = (x // 32)
    field_id = (field_grid_y * 13 + field_grid_x * 7) % 5
    
    # Base bands (Typical Sentinel-2 Level-2A surface reflectance scaled by 10,000)
    # Reflectance range: 0 - 10000 (16-bit unsigned integers)
    b02 = np.full((h, w), 550, dtype=np.float32)   # Blue
    b03 = np.full((h, w), 850, dtype=np.float32)   # Green
    b04 = np.full((h, w), 650, dtype=np.float32)   # Red
    b08 = np.full((h, w), 3400, dtype=np.float32)  # NIR (Vegetation peak)
    
    # Differentiate field types:
    # Field 0: Dense Wheat/Rye Crop (High NIR, low red)
    mask_f0 = (field_id == 0)
    b02[mask_f0] = 380; b03[mask_f0] = 720; b04[mask_f0] = 420; b08[mask_f0] = 4200
    
    # Field 1: Fallow Bare Soil (Warm soil curve)
    mask_f1 = (field_id == 1)
    b02[mask_f1] = 1250; b03[mask_f1] = 1650; b04[mask_f1] = 2100; b08[mask_f1] = 2650
    
    # Field 2: Pasture / Meadow
    mask_f2 = (field_id == 2)
    b02[mask_f2] = 420; b03[mask_f2] = 880; b04[mask_f2] = 520; b08[mask_f2] = 3600
    
    # Field 3: Dry Soil / Stubble
    mask_f3 = (field_id == 3)
    b02[mask_f3] = 1100; b03[mask_f3] = 1450; b04[mask_f3] = 1850; b08[mask_f3] = 2300

    # 2. Forest Canopy Patch (Top-Right quadrant)
    forest_mask = ((x - 190)**2 + (y - 70)**2) < 45**2
    b02[forest_mask] = 280
    b03[forest_mask] = 580
    b04[forest_mask] = 310
    b08[forest_mask] = 4800  # Extremely high NIR reflection

    # 3. Meandering Water Body / River (Flowing through scene)
    # Sine wave river corridor
    river_center = 80 + 35 * np.sin(y / 28.0) + 10 * np.cos(y / 15.0)
    river_mask = np.abs(x - river_center) < 14
    b02[river_mask] = 1350  # High blue Rayleigh / water scattering
    b03[river_mask] = 850   # Moderate green
    b04[river_mask] = 420   # Low red
    b08[river_mask] = 160   # Near-total NIR absorption in clear water

    # 4. Built-Up Village / Town Center (Bottom-Left)
    town_y, town_x = 180, 80
    town_mask = ((y - town_y)**2 + (x - town_x)**2) < 42**2
    # Individual residential buildings / rooftops (small 3x3 to 5x5 structures)
    for by in range(town_y - 35, town_y + 35, 9):
        for bx in range(town_x - 35, town_x + 35, 10):
            if ((by - town_y)**2 + (bx - town_x)**2) < 38**2:
                # Building footprint
                b02[by:by+5, bx:bx+6] = 2400  # Highly reflective tile/concrete
                b03[by:by+5, bx:bx+6] = 2600
                b04[by:by+5, bx:bx+6] = 3100  # Red clay / concrete
                b08[by:by+5, bx:bx+6] = 2900

    # 5. Road Network (Transportation corridors connecting settlements)
    # Main North-South Highway (width ~ 3 pixels = 30m at 10m res)
    road_ns = np.abs(x - (130 + (y // 60))) <= 1
    # East-West Secondary Road
    road_ew = np.abs(y - 145) <= 1
    # Diagonal Access Road to Village
    diag_road = np.abs((y - 145) - 0.7 * (x - 80)) <= 1.2
    road_mask = road_ns | road_ew | (diag_road & (x < 145) & (y > 100))
    
    # Asphalt / gravel road reflectance
    b02[road_mask] = 1400
    b03[road_mask] = 1450
    b04[road_mask] = 1500
    b08[road_mask] = 1650

    # Add realistic natural texture & sensor noise
    noise_sigma = 35.0
    b02 = np.clip(b02 + rng.normal(0, noise_sigma, (h, w)), 50, 9500).astype(np.uint16)
    b03 = np.clip(b03 + rng.normal(0, noise_sigma, (h, w)), 50, 9500).astype(np.uint16)
    b04 = np.clip(b04 + rng.normal(0, noise_sigma, (h, w)), 50, 9500).astype(np.uint16)
    b08 = np.clip(b08 + rng.normal(0, noise_sigma, (h, w)), 50, 9500).astype(np.uint16)

    # Geospatial Metadata: Poland AOI (UTM Zone 34N, EPSG:32634, 10m resolution)
    # Top-Left: Easting 635000, Northing 5785000 (Central Poland)
    crs = CRS.from_epsg(32634)
    transform = from_origin(635000.0, 5785000.0, 10.0, 10.0)

    band_dict = {
        "B02": (b02, "Blue Band 490nm (10m)"),
        "B03": (b03, "Green Band 560nm (10m)"),
        "B04": (b04, "Red Band 665nm (10m)"),
        "B08": (b08, "NIR Band 842nm (10m)")
    }

    created_paths = {}
    for band_name, (data_arr, desc) in band_dict.items():
        band_path = os.path.join(output_dir, f"{band_name}.tif")
        with rasterio.open(
            band_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=rasterio.uint16,
            crs=crs,
            transform=transform,
            compress="lzw"
        ) as dst:
            dst.write(data_arr, 1)
            dst.set_band_description(1, desc)
            dst.update_tags(data_source="synthetic illustration", georeference="illustrative, not observed")
        created_paths[band_name] = band_path

    print(f"Successfully generated Poland AOI Sentinel-2 sample dataset in: {output_dir}")
    return created_paths


if __name__ == "__main__":
    generate_poland_aoi_bands()
