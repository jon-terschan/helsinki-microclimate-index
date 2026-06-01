# this sets the global plotting style ofr everything
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import Bbox

CM_PER_INCH = 2.54


def cm_to_in(value_cm: float) -> float:
    return value_cm / CM_PER_INCH


@dataclass(frozen=True)
class FigureStyle:
    font_family: str = "Tahoma"

    # Export
    dpi_export: int = 350
    dpi_preview: int = 150
    transparent: bool = True
    export_png: bool = True
    export_pdf: bool = True
    export_svg: bool = True
    use_tight_bbox: bool = False
    pad_inches: float = 0.0

    # Standard standalone panel canvas
    panel_width_cm: float = 15
    panel_height_cm: float = 10

    # Default axes rectangle inside panel canvas: [left, bottom, width, height]
    axes_rect: tuple[float, float, float, float] = (0.15, 0.18, 0.78, 0.74)

    # Font sizes
    fs_base: float =15
    fs_tick: float = 15
    fs_axis: float = 18
    fs_title: float = 20
    fs_legend: float = 15
    fs_annotation: float = 20
    fs_panel_letter: float = 20

    # Lines and ticks
    axes_linewidth: float = 1.5
    tick_length: float = 6
    tick_width: float = 1.5
    grid_linewidth: float = 1.2
    grid_alpha: float = 0.65

    # Standard colours
    col_blue: str = "#004488"
    col_blue_fill: str = "#BBCCEE"
    col_red: str = "#BB5566"
    col_red_fill: str = "#E8BCC4"
    col_hist: str = "#DDAA33"
    col_hist_fill: str = "#E9D8A6"
    col_grey: str = "#666666"
    col_grid: str = "#D0D0D0"
    col_black: str = "#000000"


STYLE = FigureStyle()


def register_fonts(extra_font_paths: list[Path] | None = None) -> None:
    paths = [
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    ]

    if extra_font_paths:
        paths.extend(extra_font_paths)

    for path in paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def apply_style(style: FigureStyle = STYLE) -> None:
    register_fonts()

    plt.rcParams.update({
        "font.family": style.font_family,
        "font.sans-serif": [style.font_family],

        "font.size": style.fs_base,
        "axes.labelsize": style.fs_axis,
        "axes.titlesize": style.fs_title,
        "xtick.labelsize": style.fs_tick,
        "ytick.labelsize": style.fs_tick,
        "legend.fontsize": style.fs_legend,

        "axes.labelweight": "normal",
        "axes.linewidth": style.axes_linewidth,

        "figure.dpi": style.dpi_preview,
        "savefig.dpi": style.dpi_export,
        "savefig.transparent": style.transparent,
        "figure.facecolor": "none",
        "axes.facecolor": "none",

        # Important for Illustrator/Inkscape editing
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def new_panel_figure(style: FigureStyle = STYLE):
    fig = plt.figure(
        figsize=(
            cm_to_in(style.panel_width_cm),
            cm_to_in(style.panel_height_cm),
        )
    )
    fig.patch.set_alpha(0.0)
    return fig


def add_panel_axes(fig, rect=None, style: FigureStyle = STYLE):
    if rect is None:
        rect = style.axes_rect
    ax = fig.add_axes(rect)
    return ax


def style_axis(ax, style: FigureStyle = STYLE, *, grid_y=True, grid_x=False) -> None:
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        length=style.tick_length,
        width=style.tick_width,
        labelsize=style.fs_tick,
    )

    ax.grid(axis="y", visible=grid_y, color=style.col_grid,
            linewidth=style.grid_linewidth, alpha=style.grid_alpha)
    ax.grid(axis="x", visible=grid_x, color=style.col_grid,
            linewidth=style.grid_linewidth, alpha=style.grid_alpha)

    ax.set_axisbelow(True)


def set_axis_labels(ax, xlabel=None, ylabel=None, style: FigureStyle = STYLE) -> None:
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=style.fs_axis,
            fontfamily=style.font_family,
            fontweight="normal",
            color=style.col_black,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=style.fs_axis,
            fontfamily=style.font_family,
            fontweight="normal",
            color=style.col_black,
        )


def add_legend(ax, handles=None, labels=None, loc="upper right", style: FigureStyle = STYLE):
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=0.97,
        facecolor="white",
        edgecolor="#B8B8B8",
        fontsize=style.fs_legend,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.5,
        handletextpad=0.5,
    )

    try:
        legend._legend_box.align = "left"
    except Exception:
        pass

    return legend


def make_transparent(fig) -> None:
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor("none")

    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)


def save_panel(fig, out_dir: Path, basename: str, style: FigureStyle = STYLE) -> None:
    make_transparent(fig)

    save_kwargs = {
        "transparent": style.transparent,
        "facecolor": "none",
        "edgecolor": "none",
    }

    # For composition-ready exports, this should normally remain False.
    if style.use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = style.pad_inches

    out_dir.mkdir(parents=True, exist_ok=True)

    width_cm = fig.get_figwidth() * CM_PER_INCH
    height_cm = fig.get_figheight() * CM_PER_INCH

    if style.export_png:
        path = out_dir / f"{basename}.png"
        fig.savefig(path, dpi=style.dpi_export, **save_kwargs)
        print(f"[OK] wrote {path} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")

    if style.export_pdf:
        path = out_dir / f"{basename}.pdf"
        fig.savefig(path, **save_kwargs)
        print(f"[OK] wrote {path} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")

    if style.export_svg:
        path = out_dir / f"{basename}.svg"
        fig.savefig(path, **save_kwargs)
        print(f"[OK] wrote {path} ({width_cm:.2f} × {height_cm:.2f} cm canvas)")