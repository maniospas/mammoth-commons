import os
import sys
from contextlib import redirect_stdout, redirect_stderr


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def run_test(file_path):
    try:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            with open(file_path, "r") as file:
                exec(file.read(), globals())
        print(f"{file_path.ljust(60)}: {GREEN}PASS{RESET}")
        return True
    except Exception as e:
        print(f"{file_path.ljust(60)}: {RED}FAILED{RESET} - {str(e)}")
        return False


def run_tests_in_folder(path):
    success = True
    for file_name in os.listdir(path):
        if file_name.endswith(".py") and file_name.startswith("test_"):
            file_path = os.path.join(path, file_name)
            success = run_test(file_path) and success  # careful of the order (run everything on failure too)
    return success


if __name__ == "__main__":
    # monkey patch mammoth classes for tests to run quietly
    import matplotlib
    from mammoth.exports import HTML, Markdown
    HTML.show = lambda self: self.text()
    Markdown.show = lambda self: self.text()
    matplotlib.use('Agg')  # disable window visualization

    # run the actual tests
    folder_path = "tests"
    if not run_tests_in_folder(folder_path):
        sys.exit(1)  # fail github actions
