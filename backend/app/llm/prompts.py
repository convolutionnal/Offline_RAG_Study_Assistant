from typing import List, Dict, Any

SYSTEM_PROMPT = """You are an expert academic quiz generator. Your ONLY job is to generate Multiple Choice Questions grounded strictly in the provided context.

STRICT RULES:
1. Respond with ONLY a JSON array of question objects. No preamble, no explanation, no markdown fences.
2. Each object must follow EXACTLY this schema:
   {"question": str, "options": {"A": str, "B": str, "C": str, "D": str}, "correct": "A"|"B"|"C"|"D", "explanation": str}
3. The correct answer MUST be explicitly supported by the provided context.
4. Distractors (wrong options) must be plausible but clearly incorrect to someone who read the context.
5. The explanation must reference the specific concept from the context that makes the answer correct.
6. NEVER invent or assume information not present in the context.
7. If the context does not contain enough information to generate the requested number of questions, generate as many as you can.
8. If the topic is completely unrelated to the provided context or appears to be gibberish, you MUST return an empty array [].

DIFFICULTY GUIDELINES:
- easy: Direct recall of a fact stated verbatim in the context
- medium: Requires understanding a concept or process described in the context
- hard: Requires comparing, applying, or reasoning across multiple concepts in the context"""

def build_user_prompt(topic: str, chunks: list, num_questions: int, difficulty: str) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.text
        # Attempt to extract the source filename
        metadata = getattr(chunk, "metadata", {})
        source = metadata.get("source", "Unknown Document")
        
        context_blocks.append(f"[{i}] (Source: {source})\n{text}")
        
    formatted_context = "\n\n".join(context_blocks)
    
    user_prompt = f"""Please generate {num_questions} multiple-choice questions about "{topic}" at a "{difficulty}" difficulty level.

Here is the reference context:

{formatted_context}

REMINDER: You must return ONLY a valid JSON array of objects. Do not include any preambles, explanations, markdown fences, or surrounding text."""
    
    return user_prompt
