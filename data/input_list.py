"""
Input Lists for Cross-Product Generation
Maximum variation with minimum data
"""
NAMES = [
    # White-associated
    "Emily Anderson",     # Bertrand & Mullainathan (2004); Word et al. (2008)
    "Greg Thompson",      # Bertrand & Mullainathan (2004); Word et al. (2008)
    
    # Black-associated  
    "Lakisha Washington", # Bertrand & Mullainathan (2004); Word et al. (2008)
    "Jamal Jackson",      # Bertrand & Mullainathan (2004); Word et al. (2008)
    
    # Hispanic-associated
    "Carlos Martinez",    # Gaddis (2017); Word et al. (2008)
    "Maria Rodriguez",    # Gaddis (2017); Word et al. (2008)
    
    # East/Southeast Asian-associated
    "Lei Chen",           # Kang et al. (2016); Word et al. (2008)
    "Linh Nguyen",        # Kang et al. (2016); Word et al. (2008)
    
    # Middle Eastern-associated
    "Ahmed Hassan",       # Widner & Chicoine (2011); Word et al. (2008)
    "Fatima Al-Rashid",   # Widner & Chicoine (2011); Word et al. (2008)
    
    # South Asian-associated
    "Raj Sharma ",      # most common south asian/indian name
    "Aisha Patel",        # Derous et al. (2015); Word et al. (2008)
]

# Credit scores (key thresholds)
CREDIT_SCORES = [620, 640, 680, 700, 760]

# Visa status (country is always USA)
VISA_STATUS = [
    "U.S. Citizen",
    "Permanent Resident",
    "Visa Holder",
]

# Income levels (annual) - will be mapped to actual dollar amounts
INCOME = ["Low", "Medium", "High", "Ultra High"]

# Income mapping (for actual dollar values)
INCOME_MAPPING = {
    "Low": 35000,
    "Medium": 65000,
    "High": 110000,
    "Ultra High": 200000,
}

# Age 
AGE = [25, 45, 65]


LOAN_MULTIPLIERS = [1.0, 5.0]  # 1× annual income, 5× annual income