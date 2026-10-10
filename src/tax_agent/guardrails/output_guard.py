def validate_output_grounding(response_text: str) -> bool:
    """Check if the generated response appropriately cites sources or indicates missing data."""
    # Ensure the model didn't just hallucinate without acknowledging context boundaries
    if len(response_text.strip()) == 0:
        return False
    
    # Check for citation indicators or standard fallback phrases
    has_citation = "Source:" in response_text or "Publication" in response_text
    is_fallback = "do not contain enough information" in response_text.lower()
    
    return has_citation or is_fallback