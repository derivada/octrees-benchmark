import seaborn as sns

# Names of algorithms, structures and encoders
NEIGHBORS_PTR = "neighborsPtr"
NEIGHBORS = "neighbors"
NEIGHBORS_PRUNE = "neighborsPrune"
NEIGHBORS_STRUCT = "neighborsStruct"
NEIGHBORS_PCLOCT = "neighborsPclOct"
NEIGHBORS_PCLKD = "neighborsPclKD"
NEIGHBORS_UNIBN = "neighborsUnibn"
NEIGHBORS_NANOFLANN = "neighborsNanoflann"
NEIGHBORS_PICO = "neighborsPico"

LINEAR_OCTREE = "linOct"
POINTER_OCTREE = "ptrOct"
UNIBN_OCTREE = "unibnOctree"
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

# Palettes
palette_radius = sns.color_palette("tab10")
palette_knn = sns.color_palette("Set2")

# Visualization configurations
OCTREE_ENCODER = [
    {
        "params": {"octree": POINTER_OCTREE, "encoder": NO_ENCODER},
        "style": {"color": '#e1a692'},
        "display_name": r'\textit{poct\_none}'
    },
    {
        "params": {"octree": POINTER_OCTREE, "encoder": MORTON_ENCODER},
        "style": {"color": '#de6e56'},
        "display_name": r'\textit{poct\_mort}'
    },
    {
        "params": {"octree": POINTER_OCTREE, "encoder": HILBERT_ENCODER},
        "style": {"color": '#c23728'},
        "display_name": r'\textit{poct\_hilb}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "encoder": MORTON_ENCODER},
        "style": {"color": '#63bff0'},
        "display_name": r'\textit{loct\_mort}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "encoder": HILBERT_ENCODER},
        "style": {"color": '#1984c5'},
        "display_name": r'\textit{loct\_hilb}'
    }
]

OUR_RADIUS = [
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBORS_STRUCT},
        "style": {"color": palette_radius[0], "marker": 'o'},
        "display_name": r'\textit{neighborsStruct}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBORS_PRUNE},
        "style": {"color": palette_radius[1], "marker": 'o'},
        "display_name": r'\textit{neighborsPrune}'
    },
    {
        "params": {"octree": LINEAR_OCTREE, "operation": NEIGHBORS},
        "style": {"color": palette_radius[2], "marker": 'o'},
        "display_name": r'\textit{neighbors}'
    },
    {
        "params": {"octree": POINTER_OCTREE, "operation": NEIGHBORS_PTR},
        "style": {"color": palette_radius[3], "marker": 'o'},
        "display_name": r'\textit{neighborsPtr}'
    },
]

ALL_RADIUS = OUR_RADIUS + [
    {
        "params": {"octree": UNIBN_OCTREE, "operation": NEIGHBORS_UNIBN},
        "style": {"color": palette_radius[4], "marker": 'o'},
        "display_name": r'\textit{unibn Octree}'
    },
    {
        "params": {"octree": PCL_OCTREE, "operation": NEIGHBORS_PCLOCT},
        "style": {"color": palette_radius[5], "marker": 'o'},
        "display_name": r'\textit{PCL Octree}'
    },
    {
        "params": {"octree": PCL_KDTREE, "operation": NEIGHBORS_PCLKD},
        "style": {"color": palette_radius[6], "marker": 'o'},
        "display_name": r'\textit{PCL KD-tree}'
    },
    {
        "params": {"octree": NANOFLANN_KDTREE, "operation": NEIGHBORS_NANOFLANN},
        "style": {"color": palette_radius[7], "marker": 'o'},
        "display_name": r'\textit{Nanoflann KD-tree}'
    },
    {
        "params": {"octree": PICO_KDTREE, "operation": NEIGHBORS_PICO},
        "style": {"color": palette_radius[8], "marker": 'o'},
        "display_name": r'\textit{Pico KD-tree}'
    }
]

ALL_KNN = [
    {
        "params": {"octree": LINEAR_OCTREE},
        "style": {"color": palette_knn[0], "marker": 'o'},
        "display_name": r'\textit{linOctKNN}'
    },
    {
        "params": {"octree": PCL_OCTREE},
        "style": {"color": palette_knn[1], "marker": 'o'},
        "display_name": r'\textit{pclOctKNN}'
    },
    {
        "params": {"octree": PCL_KDTREE},
        "style": {"color": palette_knn[2], "marker": 'o'},
        "display_name": r'\textit{pclKdKNN}'
    },
    {
        "params": {"octree": NANOFLANN_KDTREE},
        "style": {"color": palette_knn[3], "marker": 'o'},
        "display_name": r'\textit{nanoflannKNN}'
    },
    {
        "params": {"octree": PICO_KDTREE},
        "style": {"color": palette_knn[4], "marker": 'o'},
        "display_name": r'\textit{picoKNN}'
    }
]