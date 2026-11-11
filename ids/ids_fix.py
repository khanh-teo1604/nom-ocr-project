import os

# Mục đích:
# - Xóa cột meaning đầu tiên (nếu có)
# - Bỏ cột Unicode
# - Chỉ giữ lại 2 cột: Char và Fully_Expanded_IDS

def main():
    # File đầu vào
    input_path = "ids_expanded_1.txt"

    # File đầu ra
    output_path = "ids_final_1.txt"

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.rstrip("\r\n")

            # Bỏ qua dòng trống
            if not line.strip():
                continue

            # Tách các cột bằng tab
            parts = line.split("\t")

            # Chỉ xử lý dòng có ít nhất 4 cột
            if len(parts) < 4:
                print(f"Warning: Skipping invalid line: {line}")
                continue

            # Giữ cột Char (cột 2) và Fully_Expanded_IDS (cột 4)
            character = parts[1]
            expanded_ids = parts[3]

            # Ghi ra file mới với 2 cột
            outfile.write(f"{character}\t{expanded_ids}\n")

            count += 1

    print(f"Successfully processed {count} characters.")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()