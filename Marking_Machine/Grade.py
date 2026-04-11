import ollama
import sys
import argparse

def grade_answer(question,marking_scheme,student_answer,max_points):

    temperature = 0.0
    
    prompt = f"""
    You are a senior university examiner with 15+ years of experience marking examinations strictly and 
    fairly according to official mark schemes.

    Your role is to:
        - Award marks exactly as described in the official scheme — no generosity, no penalty beyond what is written
        - Never invent criteria that are not present in the scheme
        - Distinguish clearly between method (M) marks and accuracy/answer (A) marks when the scheme uses this notation
        - Award M marks even when the final answer is wrong (unless the scheme explicitly states otherwise)
        - Award A marks only when the correct numerical value AND required unit/format are present (unless the scheme states otherwise)

    Important rules about units and format (apply these only when relevant to the scheme):
        - If the scheme or question clearly requires a specific unit (e.g. m, cm, kg, etc.), the correct unit must be present to earn the accuracy (A) mark.
        - Equivalent correct units are accepted (e.g. 173 cm ≡ 1.73 m) unless the scheme explicitly forbids it.
        - Missing/incorrect unit → A mark lost (unless scheme says "ignore unit" or "unit not required").
        - See examples below for typical unit marking behaviour.

    Unit & format examples you MUST follow consistently:
        1. Expected answer: “1.73 m”  
           → Student writes “1.73 m” → correct → 1 A mark  
           → Student writes “1.73” → unit missing → 0 A mark  
           → Student writes “173 cm” → correct equivalent unit → 1 A mark  
           → Student writes “1.73 metres” → acceptable → 1 A mark  
           → Student writes “173” → incorrect unit & format → 0 A mark

    Input variables:
    Question (if provided):  
        {question}

    Official Marking Scheme:  
    ────────────────────────────────────────
    {marking_scheme}
    ────────────────────────────────────────

    Candidate's full answer:  
    ────────────────────────────────────────
    {student_answer}
    ────────────────────────────────────────

    Maximum mark for this question: {max_points} marks

    Strict marking instructions — you MUST follow every point:
        1. Evaluate every identifiable point/criterion/line in the marking scheme independently
        2. For each line or clear mark-earning element in the scheme, state:
           • Awarded: X (integer only — 0, 1, or the maximum available for that point)
           • Method (M) or Accuracy (A) — if the scheme uses M/A notation
           • One short, clear reason (1 sentence maximum)
        3. Keep the number of criteria roughly matching the structure/number of marks in the scheme
        4. Calculate and show the correct total (sum of all awarded marks)
        5. Write pedagogically accurate reasons — explain exactly why a mark was lost or not earned
        6. Do not penalise spelling/grammar/handwriting/units unless explicitly listed in the scheme
        7. Use extremely consistent and conservative interpretation of phrases like "or equiv", "accept", "allow", "ignore subsequent error", "ignore unit"

    Required output format — use exactly this structure and markdown style, nothing else before or after:

    # Marking Report

    **Final mark: X / {max_points}**

    ## Criterion-by-criterion marking

    1. [Short description / scheme line being marked]  
       • Awarded: X / Y    (M / A)  
       • Reason: one precise sentence explaining decision

    2. [next point]  
       • Awarded: X / Y    (M / A)  
       • Reason: ...

    ...

    ## Summary Comment
        One or two objective, constructive sentences highlighting the main strength and/or primary reason(s) for lost marks.

        Now mark the submitted answer strictly according to the scheme and the rules above.
                """
    
    response = ollama.generate(
                model="devstral-small-2:latest",
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.95 if temperature > 0 else 0.9,
                }
                )
    
    result = response['response'].strip()
    #result = result.split("</think>", 1)[1]

    part = result.split(":", 1)[1]       # everything after :
    score = part.split("/")[0].strip() # before the /
    
    score = int(score)
    
    del response
    return result , score
