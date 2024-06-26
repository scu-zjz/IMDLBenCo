import os
import subprocess
import tempfile
import platform

def get_default_editor():
    """Get the system's default editor."""
    editor = os.getenv('EDITOR')
    if not editor:
        if platform.system() == 'Windows':
            editor = 'notepad'
        elif platform.system() == 'Darwin':  # macOS
            editor = 'nano'  # macOS does not have a default GUI editor
        else:  # Linux and other Unix-like systems
            editor = 'code --wait'  # 'nano' is a common text editor in Unix-like systems
    return editor

def open_file_with_editor(file_path):
    """Open the file with the default editor."""
    editor = get_default_editor()
    try:
        subprocess.run([editor, file_path])
    except Exception as e:
        print(f"Error opening file with editor {editor}: {e}")

def commit_message():
    """Simulate the Git commit message process."""
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix=".tmp") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write("# Please enter the commit message for your changes. Lines starting\n")
        temp_file.write("# with '#' will be ignored, and an empty message aborts the commit.\n")
        temp_file.write("\n")
        temp_file.write("# This is a comment line.\n")
        temp_file.flush()
        open_file_with_editor(temp_file_path)
        
        with open(temp_file_path, 'r') as f:
            commit_msg = ''.join([line for line in f if not line.startswith('#')]).strip()

        os.remove(temp_file_path)
        
        if not commit_msg:
            print("Aborting commit due to empty commit message.")
        else:
            print(f"Commit message is:\n{commit_msg}")

if __name__ == "__main__":
    commit_message()
