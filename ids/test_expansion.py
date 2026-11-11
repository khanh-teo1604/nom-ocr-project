import os
import sys

def load_expanded_ids(filepath: str) -> dict:
    """Load fully expanded IDS dictionary from file.
    
    Format: Code \\t Char \\t Original_IDS \\t Fully_Expanded_IDS
    """
    if not os.path.exists(filepath):
        print(f"Error: Expanded file not found at: {filepath}")
        print("Please run 'python3 expand_recursive_ids.py' first.")
        sys.exit(1)

    expanded_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                code, char, orig_ids, exp_ids = parts[0], parts[1], parts[2], parts[3]
                expanded_dict[char] = {
                    "code": code,
                    "original": orig_ids,
                    "expanded": exp_ids,
                }
    return expanded_dict


def test_expansion(target_chars=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "ids_expanded.txt")

    expanded_dict = load_expanded_ids(input_path)

    # Default sample characters if no custom list is passed
    if not target_chars:
        target_chars = ["夬", "明", "字", "舗", "喃", "𡨸", "𠅒"]

    print("=" * 65)
    print("IDS RECURSIVE EXPANSION VERIFICATION TEST")
    print(f"Total dictionary entries loaded: {len(expanded_dict):,}")
    print("=" * 65)

    for char in target_chars:
        if char in expanded_dict:
            data = expanded_dict[char]
            print(f"Char: {char}  ({data['code']})")
            print(f"  - Level 1 IDS : {data['original']}")
            print(f"  - Expanded IDS: {data['expanded']}")
            print("-" * 65)
        else:
            print(f"Char: {char}")
            print("  - Status      : NOT FOUND in dictionary")
            print("-" * 65)


if __name__ == "__main__":
    # If characters are passed as arguments (e.g., python3 test_expansion.py 夬 𡨸)
    user_args = sys.argv[1:]
    
    if user_args:
        # Split string if user passes a sequence of chars without spaces
        chars_to_test = []
        for arg in user_args:
            chars_to_test.extend(list(arg))
        test_expansion(chars_to_test)
    else:
        test_expansion()