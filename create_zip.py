import os
import zipfile

def create_project_zip(zip_filename="Solar_PV_Defect_Detection_Project.zip"):
    """Create a compressed ZIP archive containing all project files, code, reports, and configs."""
    print(f"Creating project zip archive: {zip_filename}...")
    
    exclude_dirs = {'__pycache__', '.git', '.pytest_cache', '.venv', 'venv'}
    exclude_files = {zip_filename}

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files or file.endswith('.pyc'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)

    print(f"ZIP archive successfully created: {zip_filename} (Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_project_zip()
