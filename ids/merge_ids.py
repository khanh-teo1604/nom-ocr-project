import os
import re

def parse_and_load(filepath, ids_dict, source_name):
    """
    Parse IDS dictionary file and load new characters into ids_dict.
    Supports CJKVI (Unicode, Char, IDS) and Yi-Bai (Char, IDS1;IDS2).
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return 0

    added_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            # Determine column format based on the first column
            if parts[0].startswith("U+") or parts[0].startswith("U-"):
                # CJKVI Format: U+XXXX \t Character \t IDS
                code_point = parts[0].strip()
                char_str = parts[1].strip()
                raw_ids = parts[2].strip() if len(parts) > 2 else char_str
            else:
                # Yi-Bai Format: Character \t IDS_Option1;IDS_Option2
                char_str = parts[0].strip()
                raw_ids = parts[1].strip()
                code_point = "" # Will generate dynamically later

            # Extract the actual first character (ignore tags like [0], (G))
            if not char_str:
                continue
            real_char = char_str[0]

            # Dynamically generate Unicode if missing
            if not code_point:
                code_point = f"U+{ord(real_char):04X}"

            # Clean IDS: Take the first option (if separated by semicolon),
            # then remove parentheses () and brackets []
            raw_ids = raw_ids.split(";")[0] 
            clean_ids = re.sub(r"\[.*?\]|\(.*?\)", "", raw_ids).strip()
            
            # Use the first block of IDS or fallback to the character itself
            clean_ids = clean_ids.split()[0] if clean_ids else real_char

            # Only add to dictionary if the character is not already present
            if real_char not in ids_dict:
                ids_dict[real_char] = (code_point, clean_ids)
                added_count += 1

    print(f"-> Loaded from {source_name:18s}: +{added_count:,} new characters added.")
    return added_count


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw_data")
    output_path = os.path.join(base_dir, "ids_merged_1.txt")

    # Order of priority: CJKVI first, then Yi-Bai lv2
    # Merge only CJKVI AND Yi-Bai lv2, ignoring lv1 and lv0 for now
    file_priority = [
        ("cjkvi_ids.txt", "CJKVI (Base)"),
        ("yibai_ids_lv2.txt", "Yi-Bai (Lv2)"),
    ]

    merged_dict = {} # Format: { char: (unicode_code, ids_sequence) }

    print("=" * 60)
    print("START MERGING IDS DICTIONARIES (CJKVI + YI-BAI)")
    print("=" * 60)

    for filename, label in file_priority:
        full_path = os.path.join(raw_dir, filename)
        parse_and_load(full_path, merged_dict, label)

    print("-" * 60)
    print(f"Total unique characters after merging: {len(merged_dict):,}")

    # Export to standard 3-column format: Unicode \t Character \t IDS
    with open(output_path, "w", encoding="utf-8") as f:
        for char, (code_point, ids_seq) in merged_dict.items():
            f.write(f"{code_point}\t{char}\t{ids_seq}\n")

    print(f"✅ Successfully exported merged file to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()