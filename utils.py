

def load_LLM(path):
    with open(path, "r") as f:
        lines = json.load(f)

    return lines