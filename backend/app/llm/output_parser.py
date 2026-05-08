import json
import re

class ParseError(Exception):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.message = message
        self.raw_text = raw_text

def extract_json(raw_text: str) -> dict | list:
    # Strip leading/trailing whitespace
    stripped_text = raw_text.strip()
    
    # Remove ```json and ``` fences if present
    text_no_fences = re.sub(r'```(?:json)?', '', stripped_text, flags=re.IGNORECASE)
    
    # Find the first { or [ character and the last } or ] character and slice to that range
    start_idx = -1
    for i, char in enumerate(text_no_fences):
        if char in ('{', '['):
            start_idx = i
            break
            
    end_idx = -1
    for i in range(len(text_no_fences) - 1, -1, -1):
        if text_no_fences[i] in ('}', ']'):
            end_idx = i
            break
            
    slice_text = ""
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        slice_text = text_no_fences[start_idx:end_idx + 1]
        
    # Attempt json.loads() on the slice
    if slice_text:
        try:
            return json.loads(slice_text)
        except json.JSONDecodeError:
            pass
            
    # If that fails, try json.loads() on the full stripped text
    try:
        return json.loads(stripped_text)
    except json.JSONDecodeError as e:
        # If both fail, raise a custom ParseError(message, raw_text=raw_text)
        raise ParseError(f"Could not parse JSON from LLM output: {str(e)}", raw_text=raw_text)
