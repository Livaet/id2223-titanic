import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass, sex, age, sibsp, parch, ticket, fare, embarked):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(ticket)
    input_list.append(fare)
    input_list.append(embarked)

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    return res[0]

# We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    #img = Image.open(requests.get(flower_url, stream=True).raw)

demo = gr.Interface(
    fn=titanic,
    title="Titanic survival Predictive Analytics",
    description="Experiment with Age, Gender, and other variables to analyze survival on the Titanic.",
    allow_flagging="never",
    inputs=[
        #gr.inputs.Number(default=1.0, label="Passenger class. Acceptable values are 1, 2, 3", precision=0),
        gr.components.Dropdown(["first", "second", "third"], type = "index"),
        gr.components.Radio(["Male","Female"], label="Gender", type="index"),
        gr.components.Slider(0, 99, value=35),
        gr.components.Checkbox(label="Travelling with sibling or spouse?"),
        gr.components.Checkbox(label="Travelling with parent or child?"),
        gr.components.Number(default=123456, label="Ticket number", precision=0),
        gr.components.Number(default=10.0, label="Fare", precision=0),
        gr.components.Dropdown(["Southampton","Cherbourg","Queenstown"], type="index")
    ],
    outputs=["label"],
)
demo.launch()

