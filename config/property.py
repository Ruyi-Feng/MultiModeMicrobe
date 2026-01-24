
def get_included_property(tag="normal"):
    if tag == "normal":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",

            "Tmin",
            "Tmax",
            "Tgrowth",
            "ToptMin",
            "ToptMax",
            "Topt.",
            "pHmin",
            "pHmax",
            "pHGrowth",
            "pHOptMin",
            "pHOptMax",
            "pHOpt.",
            "naclMin(mol/L)",
            "naclMax(mol/L)",
            "naclGrowth(mol/L)",
            "nacloptMin(mol/L)",
            "nacloptMax(mol/L)",
            "naclopt.",
            "Oxygen Tolerance",
        }
        return keys
    if tag == "pH":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",

            "pHmin",
            "pHmax",
            "pHOpt.",

        }
        return keys
    if tag == "temperature":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",

            "Tmin",
            "Tmax",
            "Topt.",
        }
        return keys
    if tag == "salinity":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",

            "naclMin(mol/L)",
            "naclMax(mol/L)",
            "naclopt.",
        }
        return keys
    if tag == "oxygen":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",

            "Oxygen Tolerance",
        }
    if tag == "opt3d":
        keys = {
            "Domain",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Species",
            "Strain",
            "pHOpt.",
            "Topt.",
            "naclopt.",

        }
    return keys
