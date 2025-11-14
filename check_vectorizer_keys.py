import pickle

for fname in ["LinearSVCTuned.pkl", "tfidfvectoizer.pkl"]:
    print(f"\n===== Inspecting {fname} =====")
    try:
        with open(fname, "rb") as f:
            obj = pickle.load(f)
        print("Type:", type(obj))
        if isinstance(obj, dict):
            print("Keys:", list(obj.keys()))
        elif hasattr(obj, "named_steps"):
            print("Pipeline steps:", list(obj.named_steps.keys()))
        else:
            print("Attributes:", [a for a in dir(obj) if not a.startswith("_")][:10])
    except Exception as e:
        print("Error reading:", e)
