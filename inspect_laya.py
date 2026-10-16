import laya
import inspect
import os

print("="*50)
print("LAYA PACKAGE DISCOVERY")
print("="*50)

# 1. Find where it's installed
pkg_dir = os.path.dirname(laya.__file__)
print(f"1. Installed at: {pkg_dir}")

# 2. List all public classes and functions
public_items = [item for item in dir(laya) if not item.startswith('_')]
print(f"2. Available items: {public_items}")

# 3. Inspect the main classes for docstrings (usage instructions)
for item in public_items:
    obj = getattr(laya, item)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        doc = inspect.getdoc(obj)
        if doc:
            print(f"\n--- {item} ---")
            print(doc)

# 4. Check for hardcoded Hugging Face model IDs in the source code
print("\n" + "="*50)
print("SEARCHING FOR HUGGING FACE MODEL IDs...")
print("="*50)

import glob
for py_file in glob.glob(os.path.join(pkg_dir, "**/*.py"), recursive=True):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Look for common HF patterns
        if "from_pretrained" in content or "hub" in content or "model_id" in content:
            print(f"\n[!] Found HF references in: {os.path.basename(py_file)}")
            # Print lines containing the keywords
            for i, line in enumerate(content.split('\n')):
                if any(kw in line for kw in ["from_pretrained", "hub", "model_id", "repo"]):
                    print(f"   Line {i+1}: {line.strip()}")