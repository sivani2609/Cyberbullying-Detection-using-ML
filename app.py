from flask import Flask, render_template, request, redirect, url_for
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    raw_model = load_pickle("LinearSVCTuned.pkl")
    print(f"[INFO] Loaded LinearSVCTuned.pkl of type: {type(raw_model)}")
except Exception as e:
    print("[ERROR] Could not load LinearSVCTuned.pkl:", e)
    raw_model = None

try:
    raw_vector = load_pickle("tfidfvectoizer.pkl")
    print(f"[INFO] Loaded tfidfvectoizer.pkl of type: {type(raw_vector)}")

    if isinstance(raw_vector, dict):
        print("[FIX] tfidfvectoizer.pkl is a vocabulary dict — rebuilding & fitting TfidfVectorizer...")
        vectorizer = TfidfVectorizer(vocabulary=raw_vector)

        dummy_texts = list(raw_vector.keys())
        vectorizer.fit(dummy_texts)
    elif hasattr(raw_vector, "transform"):
        vectorizer = raw_vector
    else:
        print("[WARNING] Invalid vectorizer format.")
        vectorizer = None
except Exception as e:
    print("[ERROR] Could not load tfidfvectoizer.pkl:", e)
    vectorizer = None


use_pipeline_directly = False
pipeline = None

if isinstance(raw_model, Pipeline):
    print("[INFO] Detected sklearn Pipeline — using pipeline directly for predictions.")
    pipeline = raw_model
    use_pipeline_directly = True
else:
    model = raw_model

print(f"[INFO] Vectorizer type: {type(vectorizer) if vectorizer else None}")
print(f"[INFO] Model type: {type(raw_model) if raw_model else None}")
print(f"[INFO] Using pipeline directly: {use_pipeline_directly}")


USERNAME = "admin"
PASSWORD = "1234"

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    if username == USERNAME and password == PASSWORD:
        return redirect(url_for("detect"))
    else:
        return render_template("login.html", error="Invalid username or password")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/predict", methods=["POST"])
def predict():
    comment = request.form.get("comment", "")

    if not comment.strip():
        return render_template("detect.html", result="Please enter a comment!")

    try:
        if use_pipeline_directly and pipeline is not None:
            pred = pipeline.predict([comment])[0]
        else:
            if vectorizer is None:
                return render_template("detect.html", result="❌ No vectorizer loaded.")
            if raw_model is None:
                return render_template("detect.html", result="❌ No model loaded.")

            X = vectorizer.transform([comment])
            pred = raw_model.predict(X)[0]

        result = "🟥 Bully Comment Detected" if int(pred) == 1 else "🟩 Non-Bully Comment"
        return render_template("detect.html", result=result, comment=comment)

    except Exception as ex:
        import traceback
        traceback.print_exc()
        return render_template("detect.html", result=f"⚠️ Error during prediction: {ex}")

if __name__ == "__main__":
    app.run(debug=True)
