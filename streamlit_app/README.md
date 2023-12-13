
# Intro
Simple demo-app to showcase the usability of the model for inference purposes.

You can access a demo app that runs inference on given audit opinion text in this [link](http://143.233.226.57:9001/).
![Demo](demo.png)

# Instructions (for local deployment)
First, make sure that you download the pre-trained model and save it in this folder.
Then you can run:

```cmd
streamlit run auditors_opinion_infer_app.py
```

And the app will be availabel at [http://localhost:8501/](http://localhost:8501/)

# Docker file
You can also use the dockerfile to deploy the model
Having downloaded the model you can run:

```cmd
docker build -t going_concern .
docker run --userns=host -p 8501:8501 -d going_concern
```

And the app will be availabel at [http://localhost:8501/](http://localhost:8501/)