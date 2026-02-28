import seaborn as sns

# Names of algorithms, structures and encoders
NEIGHBOURS_PTR = "neighboursPtr"
NEIGHBOURS = "neighbours"
NEIGHBOURS_PRUNE = "neighboursPrune"
NEIGHBOURS_STRUCT = "neighboursStruct"
NEIGHBOURS_PCLOCT = "neighboursPCLOct"
NEIGHBOURS_PCLKD = "neighboursPCLKD"
NEIGHBOURS_UNIBN = "neighboursUnibn"
NEIGHBOURS_NANOFLANN = "neighboursNanoflann"
NEIGHBOURS_PICO = "neighboursPico"

LINEAR_OCTREE = "linOct"
POINTER_OCTREE = "ptrOct"
UNIBN_OCTREE = "uniOct"
PCL_KDTREE = "pclKD"
PCL_OCTREE = "pclOct"
NANOFLANN_KDTREE = "nanoKD"
PICO_KDTREE = "picoTree"


MORTON_ENCODER = "MortonEncoder3D"
HILBERT_ENCODER = "HilbertEncoder3D"
NO_ENCODER = "NoEncoding"

# Groups of datasets to use
CLOUDS_DATASETS = {
                    "Lille_0": "Paris_Lille", 
                    "Paris_Luxembourg_6": "Paris_Lille",
                    "5080_54400": "DALES"
                }
RADII = {0.5, 1.0, 2.0, 3.0}
CLOUDS_DATASETS_HIGH_DENSITY = {
                    "bildstein_station1_xyz_intensity_rgb": "Semantic3D",
                    "sg27_station8_intensity_rgb": "Semantic3D",
                    "Speulderbos_2017_TLS": "Speulderbos"
                }
RADII_HIGH_DENSITY = {0.01, 0.05, 0.1}
ALL_CLOUDS = CLOUDS_DATASETS.copy()
ALL_CLOUDS.update(CLOUDS_DATASETS_HIGH_DENSITY)

palette = sns.color_palette("tab10")

# Visualization configurations
OCTREE_ENCODER = [
    {
        "params": {"octree": POINTER_OCTREE, "encoder": NO_ENCODER},
        "style": {"color": '#e1a692'},
        "display_name": r'\textit{ptrNone}'
    },
    {
        "params": {"octree": POINTER_OCTREE, "encoder": MORTON_ENCODER},
        "style": {"color": '#de6e56'},
        "display_name": r'\textit{ptrMort}'
    },
    {
        "params": {"octree": POINTER_OCTREE, "encoder": HILBERT_ENCODER},
        "style": {"color": '#c23728'},
        "display_name": r'\textit{ptrHilb}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "encoder": MORTON_ENCODER},
        "style": {"color": '#63bff0'},
        "display_name": r'\textit{linMort}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "encoder": HILBERT_ENCODER},
        "style": {"color": '#1984c5'},
        "display_name": r'\textit{linHilb}'
    }
]

LINEAR_OCTREE_RADIUS = [
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBOURS_STRUCT},
        "style": {"color": palette[0], "marker": 'o'},
        "display_name": r'\textit{neighboursStruct}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBOURS_PRUNE},
        "style": {"color": palette[1], "marker": 'o'},
        "display_name": r'\textit{neighboursPrune}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBOURS},
        "style": {"color": palette[2], "marker": 'o'},
        "display_name": r'\textit{neighboursLin}'
    }
]

POINTER_OCTREE_RADIUS = [
    {
        "params": {"octree": POINTER_OCTREE, "operation": NEIGHBOURS_PTR},
        "style": {"color": palette[3], "marker": 'o'},
        "display_name": r'\textit{neighboursPtr}'
    },
]

OUR_RADIUS = LINEAR_OCTREE_RADIUS + POINTER_OCTREE_RADIUS

ALL_RADIUS = OUR_RADIUS + [
    {
        "params": {"octree": UNIBN_OCTREE, "operation": NEIGHBOURS_UNIBN},
        "style": {"color": palette[4], "marker": 'o'},
        "display_name": r'\textit{neighboursUnibn}'
    },
    {
        "params": {"octree": PCL_OCTREE, "operation": NEIGHBOURS_PCLOCT},
        "style": {"color": palette[5], "marker": 'o'},
        "display_name": r'\textit{neighboursPCLOct}'
    },
    {
        "params": {"octree": PCL_KDTREE, "operation": NEIGHBOURS_PCLKD},
        "style": {"color": palette[6], "marker": 'o'},
        "display_name": r'\textit{neighboursPCLKD}'
    },
    {
        "params": {"octree": NANOFLANN_KDTREE, "operation": NEIGHBOURS_NANOFLANN},
        "style": {"color": palette[7], "marker": 'o'},
        "display_name": r'\textit{neighboursNano}'
    },
    {
        "params": {"octree": PICO_KDTREE, "operation": NEIGHBOURS_PICO},
        "style": {"color": palette[8], "marker": 'o'},
        "display_name": r'\textit{neighboursPico}'
    }
]

OTHER_RADIUS = [
    {
        "params": {"octree": UNIBN_OCTREE, "operation": NEIGHBOURS_UNIBN},
        "style": {"color": palette[4], "marker": 'o'},
        "display_name": r'\textit{neighboursUnibn}'
    },
    {
        "params": {"octree": PCL_OCTREE, "operation": NEIGHBOURS_PCLOCT},
        "style": {"color": palette[5], "marker": 'o'},
        "display_name": r'\textit{neighboursPCLOct}'
    },
    {
        "params": {"octree": PCL_KDTREE, "operation": NEIGHBOURS_PCLKD},
        "style": {"color": palette[6], "marker": 'o'},
        "display_name": r'\textit{neighboursPCLKD}'
    },
    {
        "params": {"octree": NANOFLANN_KDTREE, "operation": NEIGHBOURS_NANOFLANN},
        "style": {"color": palette[7], "marker": 'o'},
        "display_name": r'\textit{neighboursNano}'
    },
    {
        "params": {"octree": PICO_KDTREE, "operation": NEIGHBOURS_PICO},
        "style": {"color": palette[8], "marker": 'o'},
        "display_name": r'\textit{neighboursPico}'
    }
]

ALL_KNN = [
    {
        "params": {"octree": LINEAR_OCTREE},
        "style": {"color": palette[0], "marker": 'o'},
        "display_name": r'\textit{knnLinOct}'
    },
    {
        "params": {"octree": PCL_OCTREE},
        "style": {"color": palette[5], "marker": 'o'},
        "display_name": r'\textit{knnPCLOct}'
    },
    {
        "params": {"octree": PCL_KDTREE},
        "style": {"color": palette[6], "marker": 'o'},
        "display_name": r'\textit{knnPCLKD}'
    },
    {
        "params": {"octree": NANOFLANN_KDTREE},
        "style": {"color": palette[7], "marker": 'o'},
        "display_name": r'\textit{knnNano}'
    },
    {
        "params": {"octree": PICO_KDTREE},
        "style": {"color": palette[8], "marker": 'o'},
        "display_name": r'\textit{knnPico}'
    }
]