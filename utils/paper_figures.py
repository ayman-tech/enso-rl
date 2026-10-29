"""Consistent typography at the width used in the manuscript.

The current Springer manuscript places full-width figures at about 131 mm.
Size text in points *after* sizing the canvas, rather than shrinking a large
canvas (and its text) in LaTeX. Change width_mm for a different journal layout;
use the actual inclusion width, including any fractional-width figures.
"""
from pathlib import Path

PAPER_WIDTH_MM = 131.0
PAPER_FONT = 'DejaVu Sans'
PAPER_TEXT_PT = 6.5
PAPER_LABEL_PT = 7.0
PAPER_TITLE_PT = 7.5
PAPER_LEGEND_PT = 6.3
PAPER_PANEL_PT = 8.0

# Heights reserve room for readable text in small multiples.
_HEIGHT_MM = {
    'nino_power_spectrum': 82, 'mode_phase_lead': 82,
    'mye_seasonal_cycle': 64, 'action': 125,
    'forcing_lifecycle': 137, 'state_lifecycle': 137,
    'seed_sensitivity': 88, 'train_reward': 67,
    'train_diagnostics': 110, 'counterfactual': 88, 'shapley': 88,
    'xai_driver_comparison': 120, 'xai_driver_comparison_count': 120,
}


def save_paper_fig(fig, path, *, width_mm=PAPER_WIDTH_MM, height_mm=None,
                   layout=True, top=1.0, **kwargs):
    """Style and save a paper figure without altering its numerical data.

    Manual-position figures use layout=False and reserve their own label space.
    Fractional-width figures must supply their actual width_mm explicitly.
    """
    import matplotlib as mpl
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, LineCollection
    import numpy as np
    from .nb_helper import save_fig

    stem = Path(path).stem
    if height_mm is None:
        height_mm = _HEIGHT_MM.get(stem, width_mm * fig.get_figheight() / fig.get_figwidth())
    # Text is set in final-size points below; graphical marks must also be
    # adapted when a large diagnostic canvas is reduced to manuscript width.
    # Cache original sizes so a second export of the same figure is idempotent.
    if not hasattr(fig, '_paper_mark_geometry'):
        fig._paper_mark_geometry = {
            'width_inches': fig.get_figwidth(),
            'lines': [(line, line.get_markersize(), line.get_linewidth(),
                       line.get_markeredgewidth())
                      for ax in fig.axes for line in ax.lines],
            'scatter': [(artist, artist.get_sizes().copy())
                        for ax in fig.axes for artist in ax.collections
                        if isinstance(artist, PathCollection)],
            'intervals': [(artist, artist.get_linewidths().copy())
                          for ax in fig.axes for artist in ax.collections
                          if isinstance(artist, LineCollection)],
            'legend_lines': [(line, line.get_markersize(), line.get_linewidth(),
                              line.get_markeredgewidth())
                             for legend in [*fig.legends,
                                            *[ax.get_legend() for ax in fig.axes
                                              if ax.get_legend() is not None]]
                             for line in legend.findobj(Line2D)],
        }
    geometry = fig._paper_mark_geometry
    mark_scale = (width_mm / 25.4) / geometry['width_inches']
    for line, marker_size, line_width, edge_width in [*geometry['lines'],
                                                    *geometry['legend_lines']]:
        line.set_markersize(float(np.clip(marker_size * mark_scale, 2.0, 4.0)))
        line.set_linewidth(float(np.clip(line_width * mark_scale, .65, 1.5)))
        line.set_markeredgewidth(float(np.clip(edge_width * mark_scale, .4, .9)))
    for artist, sizes in geometry['scatter']:
        artist.set_sizes(np.clip(sizes * mark_scale ** 2, 4.0, 18.0))
    for artist, widths in geometry['intervals']:
        artist.set_linewidths(np.clip(widths * mark_scale, .55, 1.2))
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
    for text in fig.findobj(Text):
        text.set_fontfamily(PAPER_FONT)
        text.set_fontsize(PAPER_TEXT_PT)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        ax.tick_params(axis='both', width=.6, length=2.5)
        for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            gridline.set_linewidth(.45)
        ax.xaxis.label.set_fontsize(PAPER_LABEL_PT)
        ax.yaxis.label.set_fontsize(PAPER_LABEL_PT)
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(PAPER_TITLE_PT)
        ax.tick_params(axis='both', labelsize=PAPER_TEXT_PT)
        for text in ax.texts:
            if text.get_text().strip() in tuple('abcdefghijkl'):
                text.set_fontsize(PAPER_PANEL_PT)
                text.set_fontweight('bold')
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(PAPER_LEGEND_PT)
    for text in fig.texts:
        text.set_fontsize(PAPER_LABEL_PT)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(PAPER_LEGEND_PT)
    if fig._suptitle:
        fig._suptitle.set_fontsize(PAPER_TITLE_PT)
        if layout and top == 1.0:
            # Let constrained layout reserve title space above the axes.
            fig.suptitle(fig._suptitle.get_text(), fontsize=PAPER_TITLE_PT,
                         fontfamily=PAPER_FONT)
        else:
            fig._suptitle.set_y(.99)
    if layout:
        fig.set_layout_engine('constrained', h_pad=.045, w_pad=.045,
                              hspace=.08, wspace=.06, rect=(0, 0, 1, top))
    # A fixed-size canvas makes the exported point sizes the included point sizes.
    kwargs.pop('bbox_inches', None)
    with mpl.rc_context({'savefig.bbox': None}):
        return save_fig(fig, path, bbox_inches=None, **kwargs)
