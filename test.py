with open("file", "r") as f:
    content = f.read()
    content = content.split("\n")
with open("file1", "w") as f:
    for line in content:
        print(line)
        line = line[1:-1]
        f.write(line)
        f.write("\n")
