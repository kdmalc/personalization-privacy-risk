import seaborn as sns
import matplotlib.pyplot as plt

#_new_black = '#373737'
sns.set_theme(style='ticks', font_scale=0.75, rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 12,#9,
    'axes.labelsize': 15, #11,
    'axes.titlesize': 18, #15,
    'axes.labelpad': 2,
    'axes.linewidth': 1, #0.5,
    'axes.titlepad': 4,
    'lines.linewidth': 3,  # I have no idea what this one is, I don't think it's the plotted line but idk
    # ^ Idk actually I think my plots auto-overwrite this
    'legend.fontsize': 9,
    'legend.title_fontsize': 9,
    'xtick.labelsize': 12, #9,
    'ytick.labelsize': 12, #9,
    'xtick.major.size': 2,
    'xtick.major.pad': 1,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.pad': 1,
    'ytick.major.width': 0.5,
    'xtick.minor.size': 2,
    'xtick.minor.pad': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.size': 2,
    'ytick.minor.pad': 1,
    'ytick.minor.width': 0.5,
    
    # display axis spines
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Avoid black unless necessary
    #'text.color': _new_black,
    #'patch.edgecolor': _new_black,
    #'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    #'hatch.color': _new_black,
    #'axes.edgecolor': _new_black,
    # 'axes.titlecolor': _new_black # should fallback to text.color
    #'axes.labelcolor': _new_black,
    #'xtick.color': _new_black,
    #'ytick.color': _new_black

    # Default colormap - personal preference
    # 'image.cmap': 'inferno'
})