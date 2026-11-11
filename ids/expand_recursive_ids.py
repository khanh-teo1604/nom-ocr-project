import os
import re

# 12 standard Unicode IDS operators (U+2FF0 to U+2FFB)
IDS_OPERATORS = set([chr(code) for code in range(0x2FF0, 0x2FFC)])


def expand_ids_sequence(ids_seq: str, char2ids: dict, visited: set = None) -> str:
    """Recursively expand an IDS sequence into basic components.

    Uses cycle detection (visited set) to prevent infinite loops.
    """
    if visited is None:
        visited = set()

    result = []
    for char in ids_seq:
        # Keep operators, whitespace, or unknown characters as-is
        if char in IDS_OPERATORS or char.isspace() or char not in char2ids:
            result.append(char)
        # Stop recursion if the character forms a loop
        elif char in visited:
            result.append(char)
        else:
            new_visited = visited.copy()
            new_visited.add(char)
            expanded_sub = expand_ids_sequence(
                char2ids[char], char2ids, new_visited
            )
            result.append(expanded_sub)

    return "".join(result)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "ids_merged_1.txt")
    output_path = os.path.join(base_dir, "ids_expanded_1.txt")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    char2ids = {}
    char2code = {}

    # Load merged IDS dictionary
    skipped_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                code, char, ids_seq = (
                    parts[0].strip(),
                    parts[1].strip(),
                    parts[2].strip(),
                )
            elif len(parts) == 2:
                char, ids_seq = parts[0].strip(), parts[1].strip()
                code = f"U+{ord(char[0]):04X}"
            else:
                continue

            # Skip characters whose IDS contains '#'
            if "#" in ids_seq:
                skipped_count += 1
                continue

            char2code[char] = code
            char2ids[char] = ids_seq

    print(f"Loaded {len(char2ids):,} unique characters from {input_path}")
    print(f"Skipped {skipped_count:,} entries containing '#'.")
    print("Performing recursive IDS expansion...")

    expanded_dict = {}
    for char, ids_seq in char2ids.items():
        # Start recursion with current character in visited set to prevent self-loops
        expanded_seq = expand_ids_sequence(ids_seq, char2ids, visited={char})
        expanded_dict[char] = expanded_seq

    # Export to ids_expanded.txt (4 columns: Code \t Char \t Original_IDS \t Fully_Expanded_IDS)
    with open(output_path, "w", encoding="utf-8") as f:
        for char, ids_seq in char2ids.items():
            code = char2code.get(char, f"U+{ord(char):04X}")
            expanded_seq = expanded_dict[char]
            f.write(f"{code}\t{char}\t{ids_seq}\t{expanded_seq}\n")

    print(f"✅ Successfully exported expanded IDS dictionary to: {output_path}")

    # Display sample outputs for verification
    print("\n--- Sample Recursive Expansions ---")
    samples = ["夬", "明", "字", "舗"]
    for s in samples:
        if s in char2ids:
            print(f"Char: {s}")
            print(f"  - Level 1 IDS : {char2ids[s]}")
            print(f"  - Expanded IDS: {expanded_dict[s]}")


if __name__ == "__main__":
    main()