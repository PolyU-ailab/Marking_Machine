import os
import io
import random

import pandas as pd
import sympy as sp
from sympy.parsing.latex import parse_latex


def extract(text, start_tag, end_tag):
    """
    Extract text between tags.

    Supports both:
      &lt;\\begin{Topic_en}&gt;
    and:
      <\\begin{Topic_en}>
    """

    possible_start_tags = [
        start_tag,
        start_tag.replace("&lt;", "<").replace("&gt;", ">")
    ]

    possible_end_tags = [
        end_tag,
        end_tag.replace("&lt;", "<").replace("&gt;", ">")
    ]

    for s_tag in possible_start_tags:
        for e_tag in possible_end_tags:
            if s_tag in text and e_tag in text:
                start = text.index(s_tag) + len(s_tag)
                end = text.index(e_tag, start)
                return text[start:end].strip()

    return ""


def string_to_list(s):
    """
    Convert:
      [a, b, c]
    into:
      ["a", "b", "c"]

    This splitter is simple and works for normal step lists.
    """

    s = s.strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    items = [
        item.strip()
        for item in s.split(",")
        if item.strip()
    ]

    return items


def parse_to_DF(raw):
    raw = raw.strip()

    if not raw:
        return pd.DataFrame(columns=["Variable", "liminf", "limsup"])

    lines = raw.strip("[] \n\t").splitlines()

    clean_lines = [
        line.strip().replace("\t", "")
        for line in lines
        if line.strip() and "," in line
    ]

    csv_content = "\n".join(clean_lines)

    df = pd.read_csv(
        io.StringIO(csv_content),
        sep=",",
        skipinitialspace=True,
        engine="python"
    )

    return df


def replace_variables(text, variables):
    """
    Replace variables such as:
      *m_v*
      *t_v1*
      *t_v2*

    with randomly generated values.
    """

    if text is None:
        return text

    output = str(text)

    if variables is None or len(variables) == 0:
        return output

    for i in range(len(variables)):
        name = str(variables["Variable"][i])
        value = str(variables["values"][i])
        output = output.replace(name, value)

    return output


def is_wrapped_answer(answer_text):
    answer_text = answer_text.strip()
    return answer_text.startswith("[") and answer_text.endswith("]")


def Readquestion(question_path):
    Q = []
    questions = []
    answers = []
    marks = []
    answer_types = []
    image_paths = []

    if not os.path.isdir(question_path):
        raise FileNotFoundError(f"Question folder not found: {question_path}")

    for filename in os.listdir(question_path):
        if filename.endswith(".question"):
            full_path = os.path.join(question_path, filename)
            Q.append(full_path)

    Q.sort()

    if not Q:
        raise FileNotFoundError(f"No .question files found inside: {question_path}")

    # Page title from first .question filename.
    # Example: 2025Q1.question -> 2025Q1
    paper_title = os.path.splitext(os.path.basename(Q[0]))[0]

    topic = ""
    Variables = pd.DataFrame(columns=["Variable", "liminf", "limsup", "values"])

    for file_index in range(len(Q)):
        with open(Q[file_index], "r", encoding="utf-8") as file:
            file_content = file.read()

        if file_index == 0:
            topic = extract(
                file_content,
                "&lt;\\begin{Topic_en}&gt;",
                "&lt;\\end{Topic_en}&gt;"
            )

            variable_raw = extract(
                file_content,
                "&lt;\\begin{Variable}&gt;",
                "&lt;\\end{Variable}&gt;"
            )

            if variable_raw:
                Variables = parse_to_DF(variable_raw)
                values = []

                for i in range(len(Variables)):
                    limsup = int(Variables["limsup"][i])
                    liminf = int(Variables["liminf"][i])
                    name = str(Variables["Variable"][i])

                    dice = random.randint(liminf, limsup)
                    values.append(dice)

                    topic = topic.replace(name, str(dice))

                Variables["values"] = values
    
                
            Image_path = extract(
            file_content,
                "&lt;\\begin{Image_path}&gt;",
                "&lt;\\end{Image_path}&gt;"
            )
            
            # Keep as raw string, but still allow variable replacement if needed.
            Image_path = replace_variables(Image_path, Variables)
            image_paths.append(Image_path)

        else:
            Question_en = extract(
                file_content,
                "&lt;\\begin{Question_en}&gt;",
                "&lt;\\end{Question_en}&gt;"
            )

            Question_en = replace_variables(Question_en, Variables)
            questions.append(Question_en)

            

            Answer = extract(
                file_content,
                "&lt;\\begin{Answer}&gt;",
                "&lt;\\end{Answer}&gt;"
            )

            mark = extract(
                file_content,
                "&lt;\\begin{Mark}&gt;",
                "&lt;\\end{Mark}&gt;"
            )

            marks.append(mark)

            if is_wrapped_answer(Answer):
                answer_types.append("math_steps")

                Unit = extract(
                    file_content,
                    "&lt;\\begin{Unit}&gt;",
                    "&lt;\\end{Unit}&gt;"
                )

                Answer = replace_variables(Answer, Variables)
                Answer = string_to_list(Answer)

                # Optional: append simplified final expression.
                try:
                    eq = parse_latex(Answer[-1])
                    ans = sp.simplify(eq)
                    ans = sp.latex(ans)

                    if Unit:
                        Answer.append(f"{ans}{Unit}")
                    else:
                        Answer.append(ans)

                except Exception:
                    pass

                answers.append(Answer)

            else:
                answer_types.append("text")
                Answer = replace_variables(Answer, Variables)
                answers.append(Answer)

    return paper_title, topic, questions, answers, marks, answer_types, image_paths

