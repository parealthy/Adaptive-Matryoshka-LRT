from __future__ import annotations

import re
import signal
from typing import Dict, List, Optional

from lm_eval.utils import eval_logger

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    sympy = None
    parse_latex = None


def doc_to_text(doc: dict) -> str:
    return "Problem:\n" + doc["problem"] + "\n\nSolution:"


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction = normalize_final_answer(extract_answer(results[0]))
    reference = normalize_final_answer(str(doc["answer"]))
    return {"exact_match": int(is_equiv(prediction, reference))}


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def extract_answer(text: str) -> str:
    boxed = last_boxed_only_string(text)
    if boxed:
        return remove_boxed(boxed).strip()

    final_answer_patterns = [
        r"Final Answer:\s*(.*)",
        r"The final answer is\s*(.*)",
        r"final answer is\s*(.*)",
        r"answer is\s*(.*)",
        r"Answer:\s*(.*)",
    ]
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return cleanup_candidate(matches[-1])

    math_spans = re.findall(r"\$([^$]+)\$", text)
    if math_spans:
        return cleanup_candidate(math_spans[-1])

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        return cleanup_candidate(nonempty_lines[-1])
    return text.strip()


def cleanup_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = re.split(r"(?:\.|\n)\s*(?:I hope|Thus|Therefore|Hence)\b", candidate)[0]
    candidate = candidate.strip().rstrip(".")
    return candidate.strip()


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    ("\\left", ""),
    ("\\right", ""),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = str(final_answer).strip()
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    final_answer = final_answer.strip().rstrip(".")

    if final_answer.replace(",", "").replace("-", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


class timeout:
    def __init__(self, seconds=5, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    if x1 == x2:
        return True
    if normalize_for_string_match(x1) == normalize_for_string_match(x2):
        return True
    if sympy is None or parse_latex is None:
        return False

    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except Exception:
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False
            try:
                return bool(sympy.simplify(parsed_x1 - parsed_x2) == 0)
            except Exception:
                return bool(sympy.simplify(parsed_x1.equals(parsed_x2)))
    except Exception as exc:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {exc}")
        return False


def normalize_for_string_match(value: str) -> str:
    value = value.lower()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace(" ", "")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\\", "")
    value = value.strip().rstrip(".")
    return value
