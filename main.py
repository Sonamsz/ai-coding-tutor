
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# -----------------------------------------
# 1. SET UP GEMINI
# -----------------------------------------

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None
MODEL = "gemini-3.8-flash"


# -----------------------------------------
# 2. ASK GEMINI A QUESTION
# -----------------------------------------

def ask_gemini(prompt):
    """Send a question to Gemini and display its answer."""

    if client is None:
        print("Gemini API key not found. Check your .env file.")
        return

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=(
                "You are a friendly AI coding tutor. "
                "Explain concepts in simple words. "
                "When helpful, provide a short Python example.\n\n"
                + prompt
            ),
        )

        print("\nAI Tutor:")
        print(response.text or "No text response returned.")

    except Exception as error:
        print("\nGemini request failed.")
        print("Error type:", type(error).__name__)
        print("Error details:", str(error))


# -----------------------------------------
# 3. DEBUG PYTHON CODE
# -----------------------------------------

def debug_code():
    """Ask Gemini to explain and fix code errors."""

    print("\nPaste your Python code.")
    print("Type END on a separate line when finished.")

    lines = []

    while True:
        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    code = "\n".join(lines)

    if not code.strip():
        print("No code entered.")
        return

    prompt = f"""
Help me debug this Python code.

Please:
1. Explain what the code is supposed to do.
2. Identify any errors you can find.
3. Explain why the errors occur.
4. Provide corrected code.
5. Explain the correction in simple words.

Python code:
{code}
"""

    ask_gemini(prompt)


# -----------------------------------------
# 4. CHECK CODE BEFORE RUNNING IT
# -----------------------------------------

ALLOWED_FUNCTIONS = {
    "print", "range", "len", "sum", "min", "max",
    "abs", "round", "sorted", "enumerate", "zip",
    "int", "float", "str", "bool"
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
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
)


def validate_code(code):
    """Allow a limited set of beginner-friendly Python syntax."""

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        print("\nSyntax error:")
        print(error)
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            print(
                "\nFor safety, this runner does not allow "
                f"the syntax: {type(node).__name__}"
            )
            return False

        # Only allow direct calls to approved functions.
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                print("\nFor safety, this type of function call is blocked.")
                return False

            if node.func.id not in ALLOWED_FUNCTIONS:
                print(
                    "\nFor safety, this function is not allowed:",
                    node.func.id
                )
                return False

        # Do not allow decorators on functions.
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            print("\nFunction decorators are not allowed.")
            return False

    return True


# -----------------------------------------
# 5. RUN PYTHON CODE WITH LIMITS
# -----------------------------------------

def run_code():
    """Run simple Python code in a separate process."""

    print("\nEnter simple Python code.")
    print("Type END on a separate line when finished.")

    lines = []

    while True:
        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    code = "\n".join(lines)

    if not code.strip():
        print("No code entered.")
        return

    if len(code) > 5000:
        print("Code is too long. Keep it under 5000 characters.")
        return

    if not validate_code(code):
        print("Code was not executed.")
        return

    temp_path = None

    try:
        # Create a temporary file for the submitted code.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        print("\nRunning your code...\n")

        # -I and -S reduce access to Python environment settings
        # and automatic site-package loading.
        result = subprocess.run(
            [sys.executable, "-I", "-S", temp_path],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=tempfile.gettempdir(),
        )

        if result.stdout:
            print("Output:")
            print(result.stdout)

        if result.stderr:
            print("Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("Program finished.")
        else:
            print("Program exited with an error.")

    except subprocess.TimeoutExpired:
        print("Stopped: the program took longer than 3 seconds.")

    except Exception as error:
        print("Could not run the code.")
        print("Error type:", type(error).__name__)
        print("Error details:", str(error))

    finally:
        # Remove the temporary code file.
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except OSError:
                print("Warning: temporary file could not be removed.")


# -----------------------------------------
# 6. MAIN MENU
# -----------------------------------------

def main():
    print("=" * 40)
    print("          AI CODING TUTOR")
    print("=" * 40)

    while True:
        print("\nChoose an option:")
        print("1. Ask a Python question")
        print("2. Debug Python code")
        print("3. Run simple Python code")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            question = input("\nWhat do you want to learn? ").strip()

            if question:
                ask_gemini(question)
            else:
                print("Please enter a question.")

        elif choice == "2":
            debug_code()

        elif choice == "3":
            run_code()

        elif choice == "4":
            print("Goodbye! Keep learning Python.")
            break

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
