import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import geopandas as gpd

# Purpose: Side-by-side ERA vs CERRA visual viewer, vibecoded asf.

# load city boundary (prepared in QGIS)
gpkg_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA\helsinki_outline.gpkg"
city = gpd.read_file(gpkg_path)
line_layer = city

# enforce CRS = WGS84
if line_layer.crs is not None and line_layer.crs.to_string() != "EPSG:4326":
    line_layer = line_layer.to_crs("EPSG:4326")
minx, miny, maxx, maxy = line_layer.total_bounds

# this is vibecoded
def main():
    # -------------------------
    era_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\ERA_SUMMER_24_25_HEL.netcdf"
    cer_path = r"\\ad.helsinki.fi\home\t\terschan\Desktop\paper1\data\11.25\CERRA\CERRA_SUMMER_24_25_HEL.netcdf"
    var = "t2m"
    # -------------------------

    era = xr.open_dataset(era_path)
    cer = xr.open_dataset(cer_path)

    # convert Kelvin → Celsius
    if var in era:
        era[var] = era[var] - 273.15
    if var in cer:
        cer[var] = cer[var] - 273.15

    # match timestamps
    common = np.intersect1d(era.valid_time.values, cer.valid_time.values)
    if len(common) == 0:
        raise SystemExit("No overlapping timestamps")

    era = era.sel(valid_time=common)
    cer = cer.sel(valid_time=common)
    N = len(common)

    # ERA coords → may be 1D
    if era.longitude.ndim == 1 and era.latitude.ndim == 1:
        lonE, latE = np.meshgrid(era.longitude.values, era.latitude.values)
    else:
        lonE, latE = era.longitude.values, era.latitude.values

    # CERRA coords are 2D
    lonC = cer.longitude.values
    latC = cer.latitude.values

    # shared color scale
    vmin = float(np.nanmin([era[var].min(), cer[var].min()]))
    vmax = float(np.nanmax([era[var].max(), cer[var].max()]))

    # === FIGURE ===
    fig, (axE, axC) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.15)

    # fix extent to AOI
    axE.set_xlim(minx, maxx)
    axE.set_ylim(miny, maxy)
    axC.set_xlim(minx, maxx)
    axC.set_ylim(miny, maxy)

    # initial fields
    e0 = np.asarray(era[var].isel(valid_time=0))
    c0 = np.asarray(cer[var].isel(valid_time=0))

    meshE = axE.pcolormesh(lonE, latE, e0, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    meshC = axC.pcolormesh(lonC, latC, c0, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")

    axE.set_title("ERA")
    axC.set_title("CERRA")
    axE.set_xlabel("Lon"); axE.set_ylabel("Lat")
    axC.set_xlabel("Lon"); axC.set_ylabel("Lat")

    # ===== Add AOI outline to both panels =====
    line_layer.plot(ax=axE, edgecolor="red", linewidth=1.2)
    line_layer.plot(ax=axC, edgecolor="red", linewidth=1.2)

    # colorbar
    cbar = fig.colorbar(meshE, ax=[axE, axC], fraction=0.04, pad=0.02)
    cbar.set_label(var)

    # ===== SLIDER =====
    ax_slider = plt.axes([0.20, 0.05, 0.60, 0.03])
    slider = Slider(ax_slider, "Time", 0, N - 1, valinit=0, valstep=1)

    def update(idx):
        idx = int(idx)
        e = np.asarray(era[var].isel(valid_time=idx))
        c = np.asarray(cer[var].isel(valid_time=idx))

        meshE.set_array(e.ravel())
        meshC.set_array(c.ravel())

        axE.set_title(f"ERA — {common[idx]}")
        axC.set_title(f"CERRA — {common[idx]}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

# execute main function
if __name__ == "__main__":
    main()
