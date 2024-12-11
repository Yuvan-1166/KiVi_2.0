import __init__

for i in __init__.sentences():
    print("Client: ", i, file=open("output.txt", "a", encoding="utf-8"))
