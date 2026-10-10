def validate_tax_intent(user_query: str) -> bool:
    """Validate whether user input is relevant to tax law, filing, or IRS regulations."""
    tax_keywords = [
        "tax", "deduction", "standard", "itemize", 
        "income", "return", "irs", "filing", "dependent"
    ]
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in tax_keywords)