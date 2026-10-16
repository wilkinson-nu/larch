import numpy as np
from enum import Enum, auto


## The particle stack labels (for both truth and visible
PARTICLE_STACK_DTYPE = np.dtype([
    ("nneutron",  np.int8),
    ("nantineut", np.int8),
    ("nproton",   np.int8),    
    ("nantiprot", np.int8),
    ("npipm",     np.int8),
    ("npip",      np.int8),
    ("npim",      np.int8),
    ("npi0",      np.int8),
    ("nkapm",     np.int8),
    ("nkap",      np.int8),
    ("nkam",      np.int8),
    ("nka0",      np.int8),
    ("nem",       np.int8),
    ("nepm",      np.int8),
    ("ngamma",    np.int8),    
    ("nmuon",     np.int8),    
    ("nlambda0",  np.int8),
    ("ndeuteron", np.int8),
    ("ntritium",  np.int8),
    ("nalpha",    np.int8),
    ("nhelium3",  np.int8),
    ("nnuclfrag", np.int8),
    ("ncharged",  np.int8),
    ("ncluster",  np.int8),    
])

## These are event summary variables
EVENT_LABEL_DTYPE = np.dtype([

    ## Interaction truth
    ## TODO add Q2? W?
    ("cc",        np.bool_),
    ("enu",       np.float32),
    ("q0",        np.float32),
    ("mode",      np.int8),

    ## Summaries from the two particle stacks
    ("topology_truth",    np.int8),
    ("topology_visible",  np.int8),
    ("cctopology_truth",  np.int8),
    ("cctopology_visible",np.int8),

    ## Global deposition features
    ("edep_5mm",      np.float32),
    ("edep_10mm",     np.float32),
    ("edep_20mm",     np.float32),
    ("edep_50mm",     np.float32),
])

class Topology(Enum):

    ## Default
    NONE = -1

    ## CC topologies
    CC0pi = auto() ## 0
    CC1pi0 = auto() ## 1
    CC1pipm = auto() ## 2
    CC2pi = auto() ## 3
    CCNpi = auto() ## 4
    CCOther = auto() ## 5

    ## NC topologies
    NC0pi = auto() ## 6
    NC1pipm = auto() ## 7
    NC1pi0 = auto() ## 8
    NC2pi = auto() ## 9
    NCNpi = auto() ## 10
    NCOther = auto() ## 11
    
    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"

class Mode(Enum):

    ## Default
    NONE = -1

    ## CC modes
    CCQE = auto()
    CC2p2h = auto()
    CCRES = auto()
    CCDIS = auto()
    CCCOH = auto()

    ## NC modes
    NCQE = auto()
    NC2p2h = auto()
    NCRES = auto()
    NCDIS = auto()
    NCCOH = auto()

    ## Other
    IMD = auto()
    NUEE = auto()
    
    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"

class CCTopology(Enum):

    ## Default
    NONE = -1
    CC0pi0p = auto()                    # 0
    CC0pi1p = auto()                    # 1
    CC0piNp = auto()                    # 2

    CC1pipm0pi0_0p = auto()             # 3
    CC1pipm0pi0_1p = auto()             # 4
    CC1pipm0pi0_Np = auto()             # 5

    CCNpipm0pi0_0p = auto()             # 6
    CCNpipm0pi0_1p = auto()             # 7
    CCNpipm0pi0_Np = auto()             # 8

    CC0pipm1pi0_0p = auto()             # 9
    CC0pipm1pi0_1p = auto()             # 10
    CC0pipm1pi0_Np = auto()             # 11

    CC0pipmNpi0_0p = auto()             # 12
    CC0pipmNpi0_1p = auto()             # 13
    CC0pipmNpi0_Np = auto()             # 14

    CCmixedpi_0p = auto()               # 15
    CCmixedpi_1p = auto()               # 16
    CCmixedpi_Np = auto()               # 17

    CCOther = auto()                    # 18
    NC = auto()                         # 19
    
    ## A method to dump the list
    @classmethod
    def print_members(cls):
        for member in cls:
            print(f"{member.name}: {member.value}")

    @classmethod
    def name_from_index(cls, index):
        for member in cls:
            if member.value == index:
                return member.name
        return f"Unknown label for index {index}"
    



## TEMPORARY backwards compatibility below...
## Initial label types to store, to be clarified and then will need to be versioned (probably)
LABEL_DTYPE_EXP = np.dtype([
    ("cc",        np.bool_),
    ("topology",  np.int8),
    ("mode",      np.int8),
    ("nneutron",  np.int8),
    ("nantineut", np.int8),
    ("nproton",   np.int8),    
    ("nantiprot", np.int8),    
    ("npipm",     np.int8),
    ("npi0",      np.int8),
    ("nkapm",     np.int8),
    ("nka0",      np.int8),
    ("nem",       np.int8),
    ("nmuon",     np.int8),    
    ("nstrange",  np.int8),
    ("ncharm",    np.int8),
    ("nlambda0",  np.int8),
    ("ndeuteron", np.int8),
    ("ntritium",  np.int8),
    ("nalpha",    np.int8),
    ("nhelium3",  np.int8),
    ("nnuclfrag", np.int8),    
    ("enu",       np.float32),
    ("q0",        np.float32),
])

## To be merged later when I remake inputs...
LABEL_DTYPE_WITH_CC_CATEGORY = np.dtype(
    LABEL_DTYPE_EXP.descr
    + [("cc_category", np.int8)]
)

CC_CATEGORY_NAMES = np.array([
    "CC0pi0p",                    # 0
    "CC0pi1p",                    # 1
    "CC0piNp",                    # 2

    "CC1pipm0pi0_0p",             # 3
    "CC1pipm0pi0_1p",             # 4
    "CC1pipm0pi0_Np",             # 5

    "CCNpipm0pi0_0p",             # 6
    "CCNpipm0pi0_1p",             # 7
    "CCNpipm0pi0_Np",             # 8

    "CC0pipm1pi0_0p",             # 9
    "CC0pipm1pi0_1p",             # 10
    "CC0pipm1pi0_Np",             # 11

    "CC0pipmNpi0_0p",             # 12
    "CC0pipmNpi0_1p",             # 13
    "CC0pipmNpi0_Np",             # 14

    "CCmixedpi_0p",               # 15
    "CCmixedpi_1p",               # 16
    "CCmixedpi_Np",               # 17

    "CC-strange",                 # 18
    "Other",                      # 19
])

CC_CATEGORY_IDS = {
    name: i for i, name in enumerate(CC_CATEGORY_NAMES)
}
    

def make_cc_category(labels):
    """
    Construct mutually exclusive CC topology categories.

    CC-strange means a CC event containing at least one:
      - Lambda0
      - neutral kaon
      - charged kaon

    CC-strange takes precedence over pion/proton topology.

    Np, Npipm, and Npi0 mean multiplicity >= 2.
    """

    is_cc = np.asarray(labels["cc"], dtype=bool)
    nproton = np.asarray(labels["nproton"])
    npipm = np.asarray(labels["npipm"])
    npi0 = np.asarray(labels["npi0"])

    # Default includes NC and invalid/unclassified events.
    category = np.full(
        labels.shape,
        CC_CATEGORY_IDS["Other"],
        dtype=np.int8,
    )

    valid_counts = (
        (nproton >= 0)
        & (npipm >= 0)
        & (npi0 >= 0)
    )

    # Proton bins:
    #   0 -> 0p
    #   1 -> 1p
    #   2 -> Np (>=2)
    proton_bin = np.full(labels.shape, -1, dtype=np.int8)
    proton_bin[nproton == 0] = 0
    proton_bin[nproton == 1] = 1
    proton_bin[nproton >= 2] = 2

    # Pion topology bins:
    #   0: no pions
    #   1: exactly one charged pion, no pi0
    #   2: >=2 charged pions, no pi0
    #   3: exactly one pi0, no charged pions
    #   4: >=2 pi0, no charged pions
    #   5: both charged and neutral pions
    pion_bin = np.full(labels.shape, -1, dtype=np.int8)

    pion_bin[(npipm == 0) & (npi0 == 0)] = 0
    pion_bin[(npipm == 1) & (npi0 == 0)] = 1
    pion_bin[(npipm >= 2) & (npi0 == 0)] = 2
    pion_bin[(npipm == 0) & (npi0 == 1)] = 3
    pion_bin[(npipm == 0) & (npi0 >= 2)] = 4
    pion_bin[(npipm >= 1) & (npi0 >= 1)] = 5

    is_strange = (
        (labels["nlambda0"] > 0)
        | (labels["nka0"] > 0)
        | (labels["nkapm"] > 0)
    )

    ordinary_cc = (
        is_cc
        & valid_counts
        & ~is_strange
        & (proton_bin >= 0)
        & (pion_bin >= 0)
    )

    # Three proton categories per pion topology.
    category[ordinary_cc] = (
        3 * pion_bin[ordinary_cc]
        + proton_bin[ordinary_cc]
    ).astype(np.int8)

    # Strange CC events override ordinary topology.
    category[is_cc & is_strange] = CC_CATEGORY_IDS["CC-strange"]

    return category

def add_cc_category_field(labels):
    result = np.empty(
        labels.shape,
        dtype=LABEL_DTYPE_WITH_CC_CATEGORY,
    )

    for field_name in labels.dtype.names:
        result[field_name] = labels[field_name]

    result["cc_category"] = make_cc_category(labels)
    return result
