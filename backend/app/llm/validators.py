class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

def validate_mcq(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValidationError("Parsed item is not a dictionary.")
        
    if "question" not in parsed:
        raise ValidationError("Missing 'question' key.")
    if not isinstance(parsed["question"], str) or not parsed["question"].strip():
        raise ValidationError("'question' must be a non-empty string.")
        
    if "options" not in parsed:
        raise ValidationError("Missing 'options' key.")
    if not isinstance(parsed["options"], dict):
        raise ValidationError("'options' must be a dictionary.")
        
    expected_options = {"A", "B", "C", "D"}
    if set(parsed["options"].keys()) != expected_options:
        raise ValidationError("'options' dictionary must contain exactly keys A, B, C, and D.")
        
    for key in expected_options:
        val = parsed["options"][key]
        if not isinstance(val, str) or not val.strip():
            raise ValidationError(f"Option '{key}' must be a non-empty string.")
            
    if "correct" not in parsed:
        raise ValidationError("Missing 'correct' key.")
    if parsed["correct"] not in expected_options:
        raise ValidationError("'correct' key must be exactly one of: A, B, C, D.")
        
    if "explanation" not in parsed:
        raise ValidationError("Missing 'explanation' key.")
    if not isinstance(parsed["explanation"], str):
        raise ValidationError("'explanation' must be a string.")
    if len(parsed["explanation"].strip()) < 10:
        raise ValidationError("'explanation' must be a string with a minimum of 10 characters.")
        
    return parsed

def validate_mcq_list(parsed: list) -> list:
    if not isinstance(parsed, list):
        raise ValidationError("Parsed input must be a list.")
        
    for index, item in enumerate(parsed):
        try:
            validate_mcq(item)
        except ValidationError as e:
            raise ValidationError(f"Item at index {index} failed validation: {e.message}")
            
    return parsed
