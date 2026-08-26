import subprocess
import sys


def main():
    """Launch the Ava Streamlit Voice Application."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])


if __name__ == "__main__":
    main()
