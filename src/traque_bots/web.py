"""Interface web locale pour la demonstration et l'evaluation de jeux etiquetes."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .analysis import analyze_demo, analyze_uploaded_dataset


MAX_UPLOAD_BYTES = 64 * 1024 * 1024

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    uploaded_file = request.files.get("file")
    if uploaded_file is None:
        return jsonify(analyze_demo())

    if not uploaded_file.filename:
        return jsonify({"error": "Choisissez un fichier CSV avant de lancer l'analyse."}), 400
    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Format non pris en charge : importez un fichier .csv."}), 400

    try:
        events = pd.read_csv(
            BytesIO(uploaded_file.read()),
            dtype={"account_id": "string", "article_id": "string"},
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify({"error": "Le fichier CSV est illisible : {0}".format(error)}), 400
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Le fichier CSV est vide."}), 400

    try:
        result = analyze_uploaded_dataset(events)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Fichier trop volumineux : la limite est de 64 Mo."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
