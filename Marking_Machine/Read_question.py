import numpy as np
import pandas as pd
import io
import random
from IPython.display import display, Math
import sympy as sp
from sympy.parsing.latex import parse_latex


def extract(text,start_tag,end_tag):   
    extracted = ""
    if start_tag in text:
        if end_tag in text:
            start = text.index(start_tag) + len(start_tag)
            end   = text.index(end_tag, start)
            extracted = text[start:end].strip()
    return extracted

def string_to_list(s):
    # Remove [ and ] if present, then split by comma and strip whitespace
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    items = [item.strip() for item in s.split(',')]
    return items

def parse_to_DF(raw):
    # Remove outer [] + trim whitespace/tabs, keep only meaningful lines
    lines = raw.strip("[] \n\t").splitlines()
    clean_lines = [line.strip().replace('\t', '') for line in lines if line.strip() and ',' in line]

    csv_content = '\n'.join(clean_lines)

    df = pd.read_csv(
        io.StringIO(csv_content),
        sep=',',
        skipinitialspace=True,     # handles spaces after commas
        engine='python'            # more forgiving parser
    )

    return df

def get_question(path):
    with open(path, "r") as file:
        file_content = file.read()

    Question_start_tag = "<\\begin{Question}>"
    Question_end_tag   = "<\\end{Question}>"

    Answer_start_tag = "<\\begin{Answer}>"
    Answer_end_tag   = "<\\end{Answer}>"

    Variable_start_tag = "<\\begin{Variable}>"
    Variable_end_tag   = "<\\end{Variable}>"

    Unit_start_tag = "<\\begin{Unit}>"
    Unit_end_tag = "<\\end{Unit}>"

    
    Question = extract(file_content,Question_start_tag,Question_end_tag)
    Answers = extract(file_content,Answer_start_tag,Answer_end_tag)

    Unit = extract(file_content,Unit_start_tag,Unit_end_tag)
    #display(Math(Unit))
    Variables = extract(file_content,Variable_start_tag,Variable_end_tag)
    Variables = parse_to_DF(Variables)


    
    for i in range(len(Variables)):
        limsup = int(Variables["limsup"][i])
        liminf = int(Variables["liminf"][i])
        name = Variables["Variable"][i]
        dice = random.randint(liminf, limsup)
        Question = Question.replace(name,str(dice))
        
        Answers = Answers.replace(name,str(dice))
        
        
    Answers = string_to_list(Answers)

    eq = parse_latex(Answers[len(Answers) - 1])
    ans = sp.simplify(eq)
    ans = sp.latex(ans)
    Answers.append(f"{ans}{Unit}")
    
    #for Answer in Answers:
        #display(Math(Answer))
    
    return Question,Answers
