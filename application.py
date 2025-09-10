from flask import Flask, render_template, request
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data
            input_data = {key: [value] for key, value in request.form.items()}
            df = pd.DataFrame(input_data)

            # Run prediction
            pipeline = PredictionPipeline()
            preds = pipeline.predict(df)

            return render_template("form.html", prediction=str(preds[0]))

        except Exception as e:
            return render_template("form.html", prediction=f"Error: {str(e)}")

    return render_template("form.html", prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
