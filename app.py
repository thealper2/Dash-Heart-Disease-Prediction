import dash
from dash import dcc, html
import pickle
import numpy as np
import pandas as pd

heart_data = pd.read_csv("data/heart_cleveland_upload.csv")
X = heart_data.drop("condition", axis=1)
y = heart_data["condition"]

with open("models/extratrees.pkl", "rb") as f:
    model = pickle.load(f)

def predict_heart_disease(features):
	pred = model.predict([features])
	return pred[0]

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H1("Heart Disease Prediction"),
	html.Div([
		html.H3("Features Input"),
		html.Div([
			html.Label("Age"),
			dcc.Input(id="age-input", type="number", value=50, min=1, max=100),
			html.Br(style={"margin": "20px"}),
			html.Label("Sex"),
			dcc.RadioItems(
				id="sex-radio",
				options=[{"label": "Male", "value": 1}, {"label": "Female", "value": 0}],
				value=1
			),
			html.Br(style={"margin": "20px"}),
			html.Label("Chest Pain Type"),
			dcc.Dropdown(
				id="chest-pain-dropdown",
				options=[
					{"label": "Typical Angina", "value": 0},
					{"label": "Atypical Angina", "value": 1},
					{"label": "Non-Anginal Pain", "value": 2},
					{"label": "Asymptomatic", "value": 3}
				],
				value=0
			),
			html.Br(style={"margin": "20px"}),
			html.Label('Resting Blood Pressure'),
            dcc.Input(id='trestbps-input', type='number', value=200, min=0, max=600),
			html.Br(style={"margin": "20px"}),
			html.Label('Serum Cholestoral in mg/dl'),
            dcc.Input(id='chol-input', type='number', value=200, min=0, max=600),
			html.Br(style={"margin": "20px"}),
			html.Label("Fasting Blood Sugar > 120 mg/dl"),
			dcc.RadioItems(
				id="fbs-radio",
				options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
				value=1
			),
			html.Br(style={"margin": "20px"}),
			html.Label("Resting Electrocardiographic Results"),
			dcc.RadioItems(
				id="restecg-radio",
				options=[
					{"label": "Normal", "value": 0},
					{"label": "Having ST-T wave abnormality", "value": 1},
					{"label": "Showing probable or definite left ventricular hypertrophy by Estes criteria", "value": 2}],
				value=0
			),
			html.Br(style={"margin": "20px"}),
			html.Label("Maximum Heart Rate Achieved"),
			dcc.Input(id="thalach-input", type="number", value=90, min=0, max=180),
			html.Br(style={"margin": "20px"}),
			html.Label("Exercise Induced Angina"),
			dcc.RadioItems(
				id="exang-radio",
				options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
				value=1,
			),
			html.Br(style={"margin": "20px"}),
			html.Label("Oldpeak"),
			dcc.Input(id="oldpeak-input", type="number", value=250, min=0, max=500),
			html.Br(style={"margin": "20px"}),
			html.Label("The Slope of the Peak Exercise ST Segment"),
			dcc.RadioItems(
				id="slope-radio",
				options=[{"label": "Upsloping", "value": 0}, {"label": "Flat", "value": 1}, {"label": "Downsloping", "value": 2}],
				value=1,
			),
			html.Br(style={"margin": "20px"}),
			html.Label("Number of major vessels (0-3) colored by flourosopy"),
			dcc.Input(id="ca-input", type="number", value=0, min=0, max=3),
			html.Br(style={"margin": "20px"}),
			html.Label("Thal"),
			dcc.RadioItems(
				id="thal-radio",
				options=[{"label": "Normal", "value": 0}, {"label": "Fixed Detect", "value": 1}, {"label": "Reversable Detect", "value": 2}],
				value=0,
			),
		], style={"margin": "20px"})
	]),
	html.Button("Predict", id="predict-button", n_clicks=0),
	html.Div(id="prediction-output", style={"margin": "20px"})
])

@app.callback(
	dash.dependencies.Output("prediction-output", "children"),
    dash.dependencies.Input("predict-button", "n_clicks"),
	dash.dependencies.State("age-input", "value"),
    dash.dependencies.State("sex-radio", "value"),
    dash.dependencies.State("chest-pain-dropdown", "value"),
    dash.dependencies.State("trestbps-input", "value"),
    dash.dependencies.State("chol-input", "value"),
    dash.dependencies.State("fbs-radio", "value"),
    dash.dependencies.State("restecg-radio", "value"),
    dash.dependencies.State("thalach-input", "value"),
    dash.dependencies.State("exang-radio", "value"),
    dash.dependencies.State("oldpeak-input", "value"),
    dash.dependencies.State("slope-radio", "value"),
    dash.dependencies.State("ca-input", "value"),
    dash.dependencies.State("thal-radio", "value")
)

def update_prediction_output(n_clicks, age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
	features = [age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
	pred = predict_heart_disease(features)
	if pred == 1:
		result = "Disease"
	else:
		result = "No Disease"
	return html.Div([
		html.H4("Result:"),
		html.P(result)
	])

if __name__ == "__main__":
	app.run_server(debug=True)