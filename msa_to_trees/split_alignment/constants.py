"""Constants used throughout the split alignment module."""

# Character replacement mapping for sequence IDs (RAxML compatibility)
ILLEGAL_ID_CHARACTERS = {
    ",": "_",
    ";": "_",
    "(": "_",
    ")": "_",
    " ": "_",
    "'": "_",
    ":": "_",
}

# Regex patterns for identifying ambiguous characters by sequence type
AMBIGUOUS_NUCLEOTIDE_PATTERN = r"^[\?|N|\-|\.|\~|\!|O|X|\*]*$"
AMBIGUOUS_AMINO_ACID_PATTERN = r"^[U|X|\?'|\-|\.|\~|\*|\!]*$"
AMBIGUOUS_OTHER_PATTERN = r"^[\-|\.|\*]*$"

# Valid characters for sequence type detection (from IQ-TREE and IUPAC)
NUCLEOTIDE_CHARACTERS = (
    "R",
    "Y",
    "W",
    "S",
    "M",
    "K",
    "B",
    "H",
    "D",
    "V",
    r"\?",
    r"\-",
    r"\.",
    r"\~",
    r"\!",
    "O",
    "N",
    "X",
    r"\*",
    "A",
    "C",
    "G",
    "T",
    "U",
)

AMINO_ACID_CHARACTERS = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "O",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "B",
    "Z",
    "J",
    "U",
    "X",
    r"\?",
    r"\-",
    r"\.",
    r"\~",
    r"\*",
    r"\!",
)

# Supported alignment file formats
SUPPORTED_ALIGNMENT_FORMATS = [
    "phylip-relaxed",
    "phylip",
    "fasta",
    "nexus",
    "msf",
    "clustal",
]
