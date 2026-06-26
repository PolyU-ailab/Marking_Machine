import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock

import torch
from vllm import LLM, SamplingParams
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response

from questionreader import Readquestion
from mathverifier import MathStepVerifier, format_results


app = Flask(__name__)

# ====================== QUESTION FOLDER ======================
QUESTION_FOLDER = Path("Questions")

# ====================== GLOBAL SINGLETON MODEL ======================
llm = None
sampling_params = None
verifier = None

llm_lock = Lock()
verifier_lock = Lock()

# ====================== PAPER DATA CACHE ======================
# Important:
# This cache is NOT used to avoid re-reading forever.
# It is used so that one page refresh reads the question files once,
# then question images use the same already-loaded data.
PAPER_CACHE = {}
LATEST_CACHE_BY_SET = {}
paper_cache_lock = Lock()

MAX_PAPER_CACHE_ENTRIES = 30


def normalize_question_set(question_set):
    """
    Normalize selected question set value.
    Empty string "" means Questions/ itself.
    """
    if question_set is None:
        return ""
    return str(question_set)


def clean_latex_answer(text):
    """
    Clean submitted LaTeX answer from Live Preview.

    Removes display/inline math wrappers such as:
        \\[ ... \\]
        \\( ... \\)
        $$ ... $$

    Supports both:
        \\[x = 1\\]
        \\[x = 1
        y = 2\\]

    Also preserves multi-line math steps for MathStepVerifier.
    """
    if text is None:
        return ""

    text = str(text).strip()

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove whole-block wrappers first
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()

    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()

    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        # Remove single-line wrappers
        if line.startswith(r"\[") and line.endswith(r"\]"):
            line = line[2:-2].strip()

        elif line.startswith(r"\(") and line.endswith(r"\)"):
            line = line[2:-2].strip()

        elif line.startswith("$$") and line.endswith("$$"):
            line = line[2:-2].strip()

        # Remove leftover standalone delimiters
        if line in [r"\[", r"\]", r"\(", r"\)", "$$"]:
            continue

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def prune_paper_cache():
    """
    Prevent the temporary paper cache from growing forever.
    Keeps the newest MAX_PAPER_CACHE_ENTRIES entries.
    """
    if len(PAPER_CACHE) <= MAX_PAPER_CACHE_ENTRIES:
        return

    sorted_items = sorted(
        PAPER_CACHE.items(),
        key=lambda item: item[1].get("created_at", 0)
    )

    entries_to_remove = len(PAPER_CACHE) - MAX_PAPER_CACHE_ENTRIES

    for cache_id, entry in sorted_items[:entries_to_remove]:
        PAPER_CACHE.pop(cache_id, None)

        question_set = entry.get("question_set", "")
        if LATEST_CACHE_BY_SET.get(question_set) == cache_id:
            LATEST_CACHE_BY_SET.pop(question_set, None)


def store_paper_data(question_set, paper_data):
    """
    Store paper data for this page load and return a cache_id.

    Image routes will use this cache_id so Readquestion is not called again
    just to render images.
    """
    question_set = normalize_question_set(question_set)
    cache_id = uuid.uuid4().hex

    with paper_cache_lock:
        PAPER_CACHE[cache_id] = {
            "question_set": question_set,
            "paper_data": paper_data,
            "created_at": time.time(),
        }
        LATEST_CACHE_BY_SET[question_set] = cache_id
        prune_paper_cache()

    return cache_id


def get_cached_paper_data(cache_id, question_set):
    """
    Return cached paper data by cache_id if it exists and belongs to the
    requested question set.
    """
    question_set = normalize_question_set(question_set)

    if not cache_id:
        return None

    with paper_cache_lock:
        entry = PAPER_CACHE.get(cache_id)

        if not entry:
            return None

        if entry.get("question_set") != question_set:
            return None

        return entry.get("paper_data")


def get_latest_cached_paper_data(question_set):
    """
    Return the latest cached paper data for a question set, if available.
    """
    question_set = normalize_question_set(question_set)

    with paper_cache_lock:
        cache_id = LATEST_CACHE_BY_SET.get(question_set)

        if not cache_id:
            return None

        entry = PAPER_CACHE.get(cache_id)

        if not entry:
            return None

        return entry.get("paper_data")


def get_verifier():
    """
    Lazy load the math step verifier only once.
    """
    global verifier

    if verifier is not None:
        return verifier

    with verifier_lock:
        if verifier is None:
            print("🧮 Loading MathStepVerifier...")
            verifier = MathStepVerifier()
            print("✅ MathStepVerifier loaded.")

    return verifier


def get_llm():
    """
    Lazy load the model only once.
    """
    global llm, sampling_params

    if llm is not None:
        return llm, sampling_params

    with llm_lock:
        if llm is not None:
            return llm, sampling_params

        print("🚀 Loading Gemma-4 with vLLM. This will happen only once...")

        MODEL_NAME = "google/gemma-4-E4B-it"  # Change to E2B if VRAM limited

        llm = LLM(
            model=MODEL_NAME,
            dtype=torch.bfloat16,
            quantization="bitsandbytes",
            max_model_len=8192,
            gpu_memory_utilization=0.88,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.15,
            max_tokens=1024,
        )

        print("✅ Gemma-4 loaded successfully and ready for all users.")

    return llm, sampling_params


# ====================== SYSTEM PROMPT ======================
SYSTEM_PROMPT = """You are an expert Mathematics and Physics examiner.
Mark the student's answer strictly but fairly against the model answer.
For each question:
- Award marks for correct concepts, working steps, and final answer.
- Give partial credit where deserved.
- Point out specific errors clearly.
- Be educational in your feedback.

Respond in this exact format:
**Score:** X / Y
**Feedback:** 
[Detailed explanation]
**Strengths:** 
**Improvements:**"""


# ====================== VLM MARKING FUNCTION ======================
def call_local_vlm(vlm_prompts):
    """
    Real VLM marking using the globally loaded Gemma-4.

    If a question has Image_path LaTeX, the raw LaTeX source is also included
    in the prompt as text.
    """
    if not vlm_prompts:
        return ""

    llm_model, params = get_llm()

    messages_list = []

    for item in vlm_prompts:
        image_section = ""

        if item.get("image_path") and str(item.get("image_path")).strip():
            image_section = f"""

Question Diagram/Image LaTeX Source:
{item["image_path"]}
"""

        user_msg = f"""Question {item['q_idx'] + 1}:
{item['q_text']}
{image_section}

Maximum Marks: {item.get('max_marks', 5)}

Model Answer:
{item['model_ans']}

Student's Answer:
{item['user_ans']}

Please mark this response."""

        messages_list.append([
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_msg
            }
        ])

    print(f"🤖 Running Gemma-4 batch marking on {len(vlm_prompts)} questions...")

    outputs = llm_model.chat(messages_list, sampling_params=params)

    vlm_output = ""

    for i, output in enumerate(outputs):
        item = vlm_prompts[i]
        reply = output.outputs[0].text.strip()

        vlm_output += f"\n[VLM Assessment for Q{item['q_idx'] + 1}]\n"
        vlm_output += reply + "\n"
        vlm_output += "=" * 70 + "\n"

    return vlm_output


# ====================== QUESTION SET HELPERS ======================
def folder_has_question_files(path):
    """
    Return True when a folder directly contains one or more .question files.
    """
    try:
        return any(child.is_file() and child.suffix == ".question" for child in path.iterdir())
    except PermissionError:
        return False


def is_hidden_path(path):
    """
    Ignore hidden files/folders such as .git, .cache, etc.
    """
    return any(part.startswith(".") for part in path.parts)


def list_question_sets():
    """
    List selectable question sets inside the Questions folder.

    A folder is treated as a question set only when it directly contains
    one or more .question files.
    """
    root = QUESTION_FOLDER.resolve()

    if not root.exists():
        return []

    question_sets = []

    # Also allow Questions/ itself to be a question set if it has .question files.
    if folder_has_question_files(root):
        question_sets.append("")

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        relative_path = path.relative_to(root)

        if is_hidden_path(relative_path):
            continue

        if folder_has_question_files(path):
            question_sets.append(relative_path.as_posix())

    return sorted(question_sets)


def get_safe_browse_path(subpath=""):
    """
    Return a safe folder path under QUESTION_FOLDER for browser navigation.
    """
    root = QUESTION_FOLDER.resolve()
    target = (root / subpath).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError("Invalid folder path.")

    if not target.exists() or not target.is_dir():
        raise ValueError("Folder does not exist.")

    return root, target


def build_folder_browser(subpath=""):
    """
    Build data used by browse.html.
    """
    root, current_path = get_safe_browse_path(subpath)
    relative_path = current_path.relative_to(root)
    current_set = relative_path.as_posix() if relative_path.as_posix() != "." else ""

    try:
        children = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        children = []

    folders = []

    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue

        child_relative = child.relative_to(root).as_posix()

        folders.append({
            "name": child.name,
            "path": child_relative,
            "has_question_files": folder_has_question_files(child),
        })

    question_files = sorted(
        child.name for child in children
        if child.is_file() and child.suffix == ".question"
    )

    parent_path = None

    if current_path != root:
        parent_relative = current_path.parent.relative_to(root)
        parent_path = parent_relative.as_posix() if parent_relative.as_posix() != "." else ""

    breadcrumbs = []
    cumulative = []

    for part in relative_path.parts:
        cumulative.append(part)
        breadcrumbs.append({
            "name": part,
            "path": "/".join(cumulative),
        })

    return {
        "current_set": current_set,
        "display_path": current_set or "Questions",
        "parent_path": parent_path,
        "breadcrumbs": breadcrumbs,
        "folders": folders,
        "question_files": question_files,
        "can_open": len(question_files) > 0,
    }


def get_safe_question_path(question_set):
    """
    Convert selected question set into a safe path.

    Prevents path traversal attacks such as:
        ../../etc/passwd
    """
    question_set = normalize_question_set(question_set)

    available_sets = set(list_question_sets())

    if question_set not in available_sets:
        raise ValueError(f"Invalid question set: {question_set}")

    root = QUESTION_FOLDER.resolve()
    selected_path = (root / question_set).resolve()

    try:
        selected_path.relative_to(root)
    except ValueError:
        raise ValueError("Invalid question path.")

    if not selected_path.is_dir():
        raise ValueError(f"Question set is not a folder: {question_set}")

    return selected_path


def load_paper_data_fresh(question_set):
    """
    Load paper data fresh from disk.

    This function directly calls Readquestion().
    Use it when you intentionally want updated values:
    - page refresh
    - answer submit
    """
    question_set = normalize_question_set(question_set)

    question_path = get_safe_question_path(question_set)

    print(f"📄 Reading question set from disk: {question_set or 'Questions'}")
    print(f"📁 Full path: {question_path}")

    paper_title, topic, questions, answers, marks, answer_types, image_paths = Readquestion(
        str(question_path)
    )

    paper_data = {
        "paper_title": paper_title,
        "topic": topic,
        "questions": questions,
        "answers": answers,
        "marks": marks,
        "answer_types": answer_types,
        "image_paths": image_paths,
    }

    return paper_data


def get_paper_data_for_image(question_set, cache_id):
    """
    Get paper data for image rendering.

    Normal case:
        Use cache_id from the page, so Readquestion() is NOT called again.

    Fallback case:
        If cache is missing, use latest cached data for the question set.

    Last fallback:
        If no cache exists, read from disk once.
        This only happens for direct image access, server restart, or expired cache.
    """
    question_set = normalize_question_set(question_set)

    paper_data = get_cached_paper_data(cache_id, question_set)

    if paper_data is not None:
        return paper_data

    paper_data = get_latest_cached_paper_data(question_set)

    if paper_data is not None:
        return paper_data

    # Last fallback only.
    # This should not happen during normal page refresh image rendering.
    paper_data = load_paper_data_fresh(question_set)
    store_paper_data(question_set, paper_data)

    return paper_data


# ====================== LATEX IMAGE RENDERING ======================
def render_latex_raw_string_to_png(latex_code):
    """
    Convert a raw LaTeX string into PNG bytes.

    The Image_path value is expected to be raw LaTeX.
    """

    if latex_code is None or not str(latex_code).strip():
        raise ValueError("Empty LaTeX image source.")

    latex_code = str(latex_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "figure.tex")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_code)

        compile_proc = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                "-output-directory",
                tmpdir,
                tex_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )

        if compile_proc.returncode != 0:
            log = (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")
            raise RuntimeError("pdflatex failed:\n" + log[-4000:])

        pdf_path = os.path.join(tmpdir, "figure.pdf")

        if not os.path.exists(pdf_path):
            raise RuntimeError("LaTeX compilation did not create figure.pdf.")

        output_prefix = os.path.join(tmpdir, "figure")

        convert_proc = subprocess.run(
            [
                "pdftoppm",
                "-r",
                "200",
                "-png",
                "-singlefile",
                pdf_path,
                output_prefix,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )

        if convert_proc.returncode != 0:
            log = (convert_proc.stdout or "") + "\n" + (convert_proc.stderr or "")
            raise RuntimeError("pdftoppm failed:\n" + log[-4000:])

        png_path = output_prefix + ".png"

        if not os.path.exists(png_path):
            raise RuntimeError("PNG conversion failed. No PNG file was created.")

        with open(png_path, "rb") as f:
            png_bytes = f.read()

    return png_bytes


@app.route("/question_image/<int:q_idx>")
def render_question_image(q_idx):
    """
    Render one question's raw Image_path LaTeX string as a PNG.

    Important:
    This route normally does NOT call Readquestion().
    It uses the cache_id generated by the page refresh.
    """
    selected_set = request.args.get("set", "")
    cache_id = request.args.get("cache_id", "")

    try:
        paper_data = get_paper_data_for_image(selected_set, cache_id)
    except Exception as e:
        return Response(
            f"Could not load question set: {e}",
            status=400,
            mimetype="text/plain",
        )

    image_paths = paper_data.get("image_paths", [])

    if q_idx < 0 or q_idx >= len(image_paths):
        return Response(
            "Invalid image index.",
            status=404,
            mimetype="text/plain",
        )

    latex_code = image_paths[q_idx]

    if latex_code is None or not str(latex_code).strip():
        return Response(
            "No image for this question.",
            status=404,
            mimetype="text/plain",
        )

    try:
        png_bytes = render_latex_raw_string_to_png(latex_code)
    except Exception as e:
        return Response(
            f"Could not render LaTeX image:\n{e}",
            status=500,
            mimetype="text/plain",
        )

    return Response(png_bytes, mimetype="image/png")


# ====================== ROUTES ======================
@app.route("/browse")
@app.route("/browse/<path:subpath>")
def browse(subpath=""):
    """
    Folder navigator for choosing a question set.
    """
    try:
        browser = build_folder_browser(subpath)
    except Exception as e:
        return render_template(
            "browse.html",
            error=str(e),
            current_set="",
            display_path="Questions",
            parent_path=None,
            breadcrumbs=[],
            folders=[],
            question_files=[],
            can_open=False,
        ), 400

    return render_template("browse.html", error="", **browser)


@app.route("/")
def index():
    """
    Main question page.

    Every refresh calls Readquestion() exactly once through load_paper_data_fresh().
    Images then reuse the same paper_data through cache_id.
    """
    question_sets = list_question_sets()

    if not question_sets:
        return render_template(
            "index.html",
            paper_title="Question Set",
            topic="No question sets found in the Questions folder.",
            questions=[],
            question_sets=[],
            selected_set=""
        )

    selected_set = request.args.get("set")

    if selected_set is None:
        return redirect(url_for("browse"))

    if selected_set not in question_sets:
        selected_set = question_sets[0]

    try:
        # One disk read per refresh.
        paper_data = load_paper_data_fresh(selected_set)

        # Store the exact paper data used for this rendered page.
        page_cache_id = store_paper_data(selected_set, paper_data)

        paper_title = paper_data["paper_title"]
        topic = paper_data["topic"]
        questions = paper_data["questions"]
        marks = paper_data["marks"]
        answer_types = paper_data["answer_types"]
        image_paths = paper_data.get("image_paths", [])

    except Exception as e:
        page_cache_id = ""
        paper_title = "Question Set"
        topic = f"Error loading questions from {selected_set}: {e}"
        questions = []
        marks = []
        answer_types = []
        image_paths = []

    question_data = []

    for i, question in enumerate(questions):
        answer_type = answer_types[i] if i < len(answer_types) else "text"
        mark = marks[i] if i < len(marks) else 5
        image_path = image_paths[i] if i < len(image_paths) else ""

        has_image = bool(str(image_path).strip())

        question_data.append({
            "index": i,
            "number": i + 1,
            "question": question,
            "mark": mark,
            "answer_type": answer_type,
            "is_math_checker": answer_type == "math_steps",
            "has_image": has_image,
            "image_url": url_for(
                "render_question_image",
                q_idx=i,
                set=selected_set,
                cache_id=page_cache_id
            ) if has_image else "",
        })

    return render_template(
        "index.html",
        paper_title=paper_title,
        topic=topic,
        questions=question_data,
        question_sets=question_sets,
        selected_set=selected_set
    )


@app.route("/submit_all", methods=["POST"])
def submit_all():
    """
    Submit all answers.

    This calls Readquestion() once so marking uses the latest question values.
    """
    data = request.get_json()

    if not data:
        return jsonify({
            "ok": False,
            "error": "No JSON data received."
        }), 400

    if "submissions" not in data:
        return jsonify({
            "ok": False,
            "error": "No submissions received."
        }), 400

    selected_set = data.get("question_set", None)

    if selected_set is None:
        return jsonify({
            "ok": False,
            "error": "No question set selected."
        }), 400

    try:
        # One disk read per submit.
        paper_data = load_paper_data_fresh(selected_set)

        # Optional: update latest cache after submit too.
        store_paper_data(selected_set, paper_data)

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Could not load question set: {e}"
        }), 400

    submissions = data.get("submissions", [])

    questions = paper_data["questions"]
    answers = paper_data["answers"]
    marks = paper_data["marks"]
    answer_types = paper_data["answer_types"]
    image_paths = paper_data.get("image_paths", [])

    report_lines = [
        "=== FINAL MARKING REPORT ===",
        f"Question Set: {selected_set or 'Questions'}",
        ""
    ]

    vlm_inputs = []

    for sub in submissions:
        try:
            q_idx = int(sub.get("index"))
        except Exception:
            report_lines.append("Invalid question index received.")
            continue

        if q_idx < 0 or q_idx >= len(questions):
            report_lines.append(f"Invalid question index: {q_idx}")
            continue

        # IMPORTANT:
        # Clean Live Preview output here.
        # This removes wrappers like \[ ... \] before sending to verifier/VLM.
        raw_user_ans = sub.get("answer", "")
        user_ans = clean_latex_answer(raw_user_ans)

        q_text = questions[q_idx]
        model_ans = answers[q_idx] if q_idx < len(answers) else ""
        max_marks = marks[q_idx] if q_idx < len(marks) else 5
        image_path = image_paths[q_idx] if q_idx < len(image_paths) else ""

        q_type = answer_types[q_idx] if q_idx < len(answer_types) else "text"

        report_lines.append(f"--- Question {q_idx + 1} ({max_marks} marks) ---")
        report_lines.append(f"User Answer:\n{user_ans}\n")

        if not user_ans:
            report_lines.append("Status: Blank submission.\n")
            continue

        if q_type == "math_steps":
            latex_steps = [
                line.strip()
                for line in user_ans.splitlines()
                if line.strip()
            ]

            if len(latex_steps) < 2:
                report_lines.append(
                    "[Math Verifier]: Failed - At least two LaTeX steps required.\n"
                )
                continue

            try:
                math_verifier = get_verifier()
                results = math_verifier.verify_steps(latex_steps)
                all_valid = all(r.valid for r in results)

                report_lines.append("[Math Verifier Output]:")

                for r in results:
                    status = "✅ VALID" if r.valid else "❌ INVALID"
                    report_lines.append(
                        f"Step {r.step_index}: {status} | {r.reason}"
                    )

                # ====================== FORMAT_RESULTS OUTPUT ======================
                # This uses the format_results function imported from questionreader.py.
                # It prints the full formatted validation result into the final report.
                try:
                    formatted_results = format_results(results, show_details=True)

                    report_lines.append("")
                    report_lines.append("[Math Verifier Formatted Results]:")
                    report_lines.append(str(formatted_results))
                    report_lines.append("")

                except Exception as e:
                    report_lines.append("")
                    report_lines.append(
                        f"[Math Verifier Formatted Results Error]: {str(e)}"
                    )
                    report_lines.append("")

                if all_valid:
                    report_lines.append(
                        "→ All steps valid. Sending to Gemma-4...\n"
                    )

                    vlm_inputs.append({
                        "q_idx": q_idx,
                        "q_text": q_text,
                        "user_ans": user_ans,
                        "model_ans": model_ans,
                        "max_marks": max_marks,
                        "image_path": image_path,
                    })
                else:
                    report_lines.append(
                        "→ Invalid steps detected. Skipped VLM marking.\n"
                    )

            except Exception as e:
                report_lines.append(f"[Math Verifier Error]: {str(e)}\n")

        else:
            vlm_inputs.append({
                "q_idx": q_idx,
                "q_text": q_text,
                "user_ans": user_ans,
                "model_ans": model_ans,
                "max_marks": max_marks,
                "image_path": image_path,
            })

    # ====================== RUN GEMMA-4 MARKING ======================
    if vlm_inputs:
        report_lines.append("\n=== GEMMA-4 VLM EVALUATION ===\n")

        try:
            vlm_report = call_local_vlm(vlm_inputs)
            report_lines.append(vlm_report)

        except Exception as e:
            report_lines.append(f"Error during Gemma-4 inference: {str(e)}")

    final_report = "\n".join(report_lines)

    return jsonify({
        "ok": True,
        "report": final_report
    })


if __name__ == "__main__":
    # Optional preload.
    # If you want faster first marking, keep this.
    # If you want the website to start faster, comment it out.
    get_llm()

    app.run(
        host="0.0.0.0",
        port=5005,
        debug=False
    )
