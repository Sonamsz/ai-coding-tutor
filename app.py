
import ast
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Coding Tutor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Use the model that worked in your previous test.
MODEL = "gemini-3.8-flash"


# ==========================================
# 2. CUSTOM INTERFACE STYLING
# ==========================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 14px;
        margin-bottom: 10px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 3. GEMINI AI FUNCTION WITH RETRIES
# ==========================================

def ask_gemini(prompt):
    """Ask Gemini a question and retry temporary errors."""

    if client is None:
        return (
            "Gemini API key not found.\n\n"
            "Check that your .env file contains GEMINI_API_KEY."
        )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=(
                    "You are a friendly AI coding tutor. "
                    "Explain programming concepts in simple words. "
                    "When debugging code, identify errors, explain "
                    "why they occur, and provide corrected code "
                    "when appropriate.\n\n"
                    + prompt
                ),
            )

            return response.text or "Gemini returned no text response."

        except Exception as error:
            details = str(error)
            error_type = type(error).__name__

            temporary_error = any(
                marker in details.upper()
                for marker in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                )
            )

            if temporary_error and attempt < 2:
                # Wait 2 seconds, then 4 seconds.
                time.sleep(2 ** (attempt + 1))
                continue

            if "503" in details or "UNAVAILABLE" in details:
                return (
                    "Gemini is temporarily experiencing high demand. "
                    "Please try again later.\n\n"
                    f"Error details: {details}"
                )

            if "429" in details or "RESOURCE_EXHAUSTED" in details:
                return (
                    "Gemini's usage limit may have been reached. "
                    "Check your API usage and quota, then try later.\n\n"
                    f"Error details: {details}"
                )

            return (
                f"Request failed ({error_type}).\n\n"
                f"Error details: {details}"
            )

    return "The request could not be completed. Please try again later."


# ==========================================
# 4. BASIC PYTHON CODE VALIDATION
# ==========================================

ALLOWED_FUNCTIONS = {
    "print",
    "range",
    "len",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "sorted",
    "enumerate",
    "zip",
    "int",
    "float",
    "str",
    "bool",
}

ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Call,
    ast.Load,
    ast.Store,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.IfExp,
)


def validate_code(code):
    """Allow a limited set of beginner-friendly Python syntax."""

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"Syntax error: {error}"

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return (
                False,
                f"Blocked for safety: {type(node).__name__}",
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return (
                    False,
                    "Only approved direct function calls are allowed.",
                )

            if node.func.id not in ALLOWED_FUNCTIONS:
                return (
                    False,
                    f"Function '{node.func.id}' is not allowed.",
                )

    return True, "Code passed the basic checks."


# ==========================================
# 5. PYTHON CODE RUNNER
# ==========================================

def run_python(code):
    """Run simple Python code in a separate process."""

    if not code.strip():
        return "Please enter some Python code."

    if len(code) > 5000:
        return "Code is too long. Keep it under 5000 characters."

    valid, message = validate_code(code)

    if not valid:
        return f"{message}\n\nCode was not executed."

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=tempfile.gettempdir(),
        )

        output = ""

        if result.stdout:
            output += result.stdout

        if result.stderr:
            output += "\nErrors:\n" + result.stderr

        if not output:
            output = "Program finished without producing output."

        if result.returncode == 0:
            output += "\n\nProgram finished successfully."
        else:
            output += "\n\nProgram exited with an error."

        return output

    except subprocess.TimeoutExpired:
        return "Execution stopped because the program exceeded 3 seconds."

    except Exception as error:
        return (
            f"Execution failed: {type(error).__name__}: {error}"
        )

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.title("💻 AI Coding Tutor")
    st.caption("Your personal Python learning assistant")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "💬 Ask AI",
            "🐛 Debug Code",
            "▶️ Run Code",
        ],
    )

    st.divider()

    st.markdown("### Your learning toolkit")
    st.markdown("- Learn Python concepts")
    st.markdown("- Understand coding errors")
    st.markdown("- Practise with simple programs")

    st.divider()

    st.caption("Powered by Gemini AI and Python")


# ==========================================
# 7. HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown(
        '<p class="main-title">Welcome to AI Coding Tutor 👋</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtitle">'
        "Learn Python step by step with your AI assistant."
        "</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "Choose a feature below to begin learning, debugging, "
        "or practising Python."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>💬 Ask AI</h3>
                <p>Ask Python questions and get simple explanations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🐛 Debug Code</h3>
                <p>Find errors and learn how to correct your code.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>▶️ Run Code</h3>
                <p>Execute supported beginner-level Python programs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Try a Python question")

    home_question = st.text_input(
        "What would you like to learn?",
        placeholder="Example: What is a Python variable?",
        key="home_question",
    )

    if st.button("Ask my tutor", type="primary"):
        if home_question.strip():
            with st.spinner("Your tutor is thinking..."):
                answer = ask_gemini(home_question)

            st.markdown("### AI Tutor's Answer")
            st.markdown(answer)
        else:
            st.warning("Please enter a question.")


# ==========================================
# 8. ASK AI PAGE
# ==========================================

elif page == "💬 Ask AI":

    st.title("💬 Ask AI")
    st.write(
        "Ask any Python question. Your tutor will explain it "
        "in simple language and provide examples."
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: Explain Python lists with a simple example."
        ),
        height=140,
        key="ask_question",
    )

    if st.button("✨ Ask AI", type="primary"):
        if question.strip():
            with st.spinner("Thinking..."):
                answer = ask_gemini(question)

            st.markdown("### AI Tutor's Answer")
            st.markdown(answer)
        else:
            st.warning("Please enter a question.")


# ==========================================
# 9. DEBUG CODE PAGE
# ==========================================

elif page == "🐛 Debug Code":

    st.title("🐛 Python Debugging Assistant")

    st.write(
        "Paste your Python code. The AI will identify possible "
        "errors, explain why they occur, and suggest corrections."
    )

    debug_code = st.text_area(
        "Paste your Python code",
        value='print("Hello"',
        height=250,
        key="debug_input",
    )

    if st.button("🔍 Debug My Code", type="primary"):
        if debug_code.strip():
            prompt = f"""
Debug this Python code.

Please:
1. Explain what the code is trying to do.
2. Identify the errors you can find.
3. Explain why those errors happen.
4. Provide corrected code.
5. Explain the correction in simple words.

Do not claim to have executed the code. Only analyse it.

Python code:
{debug_code}
"""

            with st.spinner("Analysing your code..."):
                answer = ask_gemini(prompt)

            st.markdown("### Debugging Results")
            st.markdown(answer)
        else:
            st.warning("Please paste some code first.")


# ==========================================
# 10. RUN CODE PAGE
# ==========================================

elif page == "▶️ Run Code":

    st.title("▶️ Python Code Runner")

    st.write(
        "Write a small Python program and see its output."
    )

    st.warning(
        "Safety notice: this beginner runner blocks many operations, "
        "but it is NOT a secure sandbox. Only run code you understand "
        "and trust."
    )

    example_code = (
        'fruits = ["apple", "banana", "cherry"]\n\n'
        "for fruit in fruits:\n"
        "    print(fruit)"
    )

    code = st.text_area(
        "Write your Python code",
        value=example_code,
        height=260,
        key="run_input",
    )

    if st.button("▶ Run Python", type="primary"):
        with st.spinner("Running your code..."):
            output = run_python(code)

        st.markdown("### Console Output")
        st.code(output, language="text")


# ==========================================
# END OF APPLICATION
# ==========================================
