import os

def delete_matching_files(source_dir, compare_dir):
    source_files = os.listdir(source_dir)
    compare_files = set(os.listdir(compare_dir))  # faster lookup

    deleted = []

    for filename in source_files:
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path) and filename in compare_files:
            os.remove(source_path)
            deleted.append(filename)

    print(f"✅ Deleted {len(deleted)} file(s) from '{source_dir}':")
    for f in deleted:
        print(f" - {f}")

# Example usage:
delete_matching_files("input2", "sort_input2")
