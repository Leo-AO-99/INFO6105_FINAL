def remove_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    unique_lines = list(dict.fromkeys(lines))  # Preserve order and remove duplicates

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

remove_duplicates('sentence.txt')
