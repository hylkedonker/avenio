from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# target_genes = [
#     "ABCC5",
#     "ABCG2",
#     "ACTN2",
#     "ADAMTS12",
#     "ADAMTS16",
#     "ARFGEF1",
#     "ASTN1",
#     "ASTN2",
#     "AVPR1A",
#     "BCHE",
#     "BPIFB4",
#     "BRINP2",
#     "BRINP3",
#     "C6",
#     "C6ORF118",
#     "CA10",
#     "CACNA1E",
#     "CDH12",
#     "CDH18",
#     "CDH8",
#     "CDH9",
#     "CDKN2A",
#     "CHRM2",
#     "CNTN5",
#     "CNTNAP2",
#     "CPXCR1",
#     "CPZ",
#     "CRMP1",
#     "CSMD1",
#     "CSMD3",
#     "CTNNB1",
#     "CTNND2",
#     "CYBB",
#     "DCAF12L1",
#     "DCAF12L2",
#     "DCAF4L2",
#     "DCLK1",
#     "DCSTAMP",
#     "DDI1",
#     "DLGAP2",
#     "DMD",
#     "DNTTIP1",
#     "DOCK3",
#     "DSC3",
#     "DSCAM",
#     "EDSCAM",
#     "EGFLAM",
#     "EPHA5",
#     "EPHA6",
#     "EYS",
#     "FAM135B",
#     "FAM151A",
#     "FAM71B",
#     "FAT1",
#     "FBN2",
#     "FBXL7",
#     "FBXW7",
#     "FCRL5",
#     "FOXG1",
#     "FRYL",
#     "GBA3",
#     "GBP7",
#     "GJA8",
#     "GPR139",
#     "GRIA2",
#     "GRIK3",
#     "GRIN2B",
#     "GRIN3B",
#     "GRM1",
#     "GRM5",
#     "GRM8",
#     "GSX1",
#     "HACD1",
#     "HCN1",
#     "HCRTR2",
#     "HEBP1",
#     "HECW1",
#     "HS3ST4",
#     "HS3ST5",
#     "HTR1A",
#     "HTR1E",
#     "HTR2C",
#     "IFI16",
#     "IL7R",
#     "INSL3",
#     "ITGA10",
#     "ITSN1",
#     "KCNA5",
#     "KCNB2",
#     "KCNC2",
#     "KCNJ3",
#     "KCTD8",
#     "KEAP1",
#     "KIAA1211",
#     "KIF17",
#     "KIF19",
#     "KLHL31",
#     "KPRP",
#     "LPPR4",
#     "LRFN5",
#     "LRP1B",
#     "LRRC7",
#     "LRRTM1",
#     "LRRTM4",
#     "LTBP4",
#     "MAP2",
#     "MAP7D3",
#     "MKRN3",
#     "MMP16",
#     "MTX1",
#     "MYH7",
#     "MYT1L",
#     "NAV3",
#     "NEUROD4",
#     "NFE2L2",
#     "NLGN4X",
#     "NLRP3",
#     "NMUR1",
#     "NOL4",
#     "NPAP1",
#     "NR0B1",
#     "NRXN1",
#     "NXPH4",
#     "NYAP2",
#     "OPRD1",
#     "P2RY10",
#     "PAX6",
#     "PCDH15",
#     "PDYN",
#     "PDZRN3",
#     "PGK2",
#     "PHACTR1",
#     "PIK3CA",
#     "PIK3CG",
#     "PKHD1L1",
#     "POLE",
#     "POM121L12",
#     "PREX1",
#     "RALYL",
#     "RFX5",
#     "RIN3",
#     "RNASE3",
#     "ROBO2",
#     "SEMA5B",
#     "SLC18A3",
#     "SLC39A12",
#     "SLC6A5",
#     "SLC8A1",
#     "SLITRK1",
#     "SLITRK4",
#     "SLITRK5",
#     "SLPI",
#     "SMAD4",
#     "SOX9",
#     "SPTA1",
#     "ST6GALNAC3",
#     "STK11",
#     "SV2A",
#     "T",
#     "THSD7A",
#     "TIAM1",
#     "TMEM200A",
#     "TNFRSF21",
#     "TNN",
#     "TNR",
#     "TRHDE",
#     "TRIM58",
#     "TRPS1",
#     "UGT3A2",
#     "USH2A",
#     "USP29",
#     "VPS13B",
#     "WBSCR17",
#     "WIPF1",
#     "WSCD2",
#     "ZC3H12A",
#     "ZFPM2",
#     "ZIC1",
#     "ZIC4",
#     "ZNF521",
#     "ZSCAN1",
# ]

target_genes = [
    "ABL1",
    "AKT1",
    "AKT2",
    "ALK",
    "APC",
    "AR",
    "ARAF",
    "BRAF",
    "BRCA1",
    "BRCA2",
    "CCND1",
    "CCND2",
    "CCND3",
    "CD274",
    "CDK4",
    "CDK6",
    "CDKN2A",
    "CSF1R",
    "CTNNB1",
    "DDR2",
    "DPYD",
    "EGFR",
    "ERBB2",
    "ESR1",
    "EZH2",
    "FBXW7",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "FLT1",
    "FLT3",
    "FLT4",
    "GATA3",
    "GNA11",
    "GNAQ",
    "GNAS",
    "IDH1",
    "IDH2",
    "JAK2",
    "JAK3",
    "KDR",
    "KEAP1",
    "KIT",
    "KRAS",
    "MAP2K1",
    "MAP2K2",
    "MET",
    "MLH1",
    "MSH2",
    "MSH6",
    "MTOR",
    "NF2",
    "NFE2L2",
    "NRAS",
    "NTRK1",
    "PDCD1LG2",
    "PDGFRA",
    "PDGFRB",
    "PIK3CA",
    "PIK3R1",
    "PMS2",
    "PTCH1",
    "PTEN",
    "RAF1",
    "RB1",
    "RET",
    "RNF43",
    "ROS1",
    "SMAD4",
    "SMO",
    "STK11",
    "TERT",
    "TP53",
    "TSC1",
    "TSC2",
    "UGT1A1",
    "VHL",
]

# Phenotype features that serve as input for the model.
clinical_features = [
    "gender",
    "Age",
    "stage",
    "therapyline",
    "smokingstatus",
    "Systemischetherapie",
    "histology_grouped",
    "lymfmeta",
    "brainmeta",
    "adrenalmeta",
    "livermeta",
    "lungmeta",
    "skeletonmeta",
]

# From those listed above, the following columns are categorical (not counting
# the labels).
categorical_phenotypes = [
    "gender",
    "stage",
    "therapyline",
    "smokingstatus",
    "Systemischetherapie",
    "histology_grouped",
    "lymfmeta",
    "brainmeta",
    "adrenalmeta",
    "livermeta",
    "lungmeta",
    "skeletonmeta",
]


def get_hyper_param_grid(model) -> dict:
    """
    Get parameter grid for hyper parameter tuning.
    """
    filter_params = {}

    prefix = ""
    if isinstance(model, Pipeline):
        prefix = "estimator__"
        if "statistical_filter" in model.named_steps:
            filter_params.update({"statistical_filter__alpha": [0.05, 0.1, 0.2, 0.4]})
        model = model.named_steps["estimator"]

    if isinstance(model, LogisticRegression):
        filter_params.update(
            {
                f"{prefix}C": [
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.175,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.5,
                    2.0,
                    4.0,
                ]
            }
        )
    elif isinstance(model, DecisionTreeClassifier):
        filter_params.update(
            {
                f"{prefix}max_depth": [2, 3, 5, 7, 10, 15, 20],
                f"{prefix}criterion": ["gini", "entropy"],
            }
        )
    elif isinstance(model, RandomForestClassifier):
        filter_params.update(
            {
                f"{prefix}n_estimators": [15, 30, 50, 100],
                f"{prefix}max_depth": [2, 3, 5, 7, 10, 15, None],
                f"{prefix}class_weight": ["balanced", "balanced_subsample"],
            }
        )
    elif isinstance(model, GradientBoostingClassifier):
        filter_params.update(
            {
                f"{prefix}n_estimators": [15, 30, 50, 100],
                f"{prefix}learning_rate": [0.025, 0.05, 0.1, 0.2],
                f"{prefix}max_depth": [2, 3, 5, 7],
            }
        )
    elif isinstance(model, KNeighborsClassifier):
        filter_params.update(
            {
                f"{prefix}n_neighbors": [2, 3, 4, 6, 8, 12, 20],
                f"{prefix}weights": ["uniform", "distance"],
                f"{prefix}p": [1, 2, 3],
            }
        )
    elif isinstance(model, SVC):
        filter_params.update(
            {
                f"{prefix}C": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                f"{prefix}kernel": ["linear", "poly", "rbf", "sigmoid"],
                f"{prefix}gamma": ["auto", "scale"],
            }
        )
    elif isinstance(model, CategoricalNB):
        filter_params.update({f"{prefix}alpha": [0.125, 0.25, 0.5, 1.0, 2.0]})

    return filter_params
