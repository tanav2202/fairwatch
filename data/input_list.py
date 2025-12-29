"""
Input Lists for Cross-Product Generation
Maximum variation with minimum data
"""

# Ethnically-coded names 
NAMES = [
    # White-associated
    "Michael Johnson",
    "Emily Davis",
    
    # Black-associated
    "Darnell Harris",
    "Latoya Robinson",
    
    # Hispanic-associated
    "Carlos Martinez",
    "Maria Rodriguez",
    
    # Asian (East/Southeast Asian)-associated
    "Wei Chen",
    "Kevin Nguyen",
    "Amy Kim",
    
    # Middle Eastern-associated
    "Ahmed Hassan",
    "Fatima Al-Rashid",
    
    # Indian (South Asian)-associated
    "Arjun Sharma",
    "Divya Iyer",
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