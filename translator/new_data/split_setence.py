
if __name__ == "__main__":
    src_list = []
    tgt_list = []
    with open("sentence.txt", "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if "&" not in line:
                continue
            src, tgt = line.split("&")
            src_list.append(src)
            tgt_list.append(tgt)

    with open("src.txt", "w", encoding="utf-8") as f:
        for src in src_list:
            f.write(src + "\n")
    with open("tgt.txt", "w", encoding="utf-8") as f:
        for tgt in tgt_list:
            f.write(tgt + "\n")
