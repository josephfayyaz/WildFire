import os

def delete_tif_files_from_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        tif_path = line.strip()
        if tif_path:  # Skip empty lines
            if os.path.exists(tif_path):
                try:
                    os.remove(tif_path)
                    print(f"Deleted: {tif_path}")
                except Exception as e:
                    print(f"Failed to delete {tif_path}: {e}")
            else:
                print(f"File not found: {tif_path}")

# Example usage:
# Replace this with the actual file you're using each time
file_to_process = 'src/fire/eliminateNoData/ToBeDeletedPre.txt'
delete_tif_files_from_list(file_to_process)
