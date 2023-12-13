# Use an official Python runtime as a parent image
FROM python:3.9-slim

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN pip install -r requirements.txt

COPY . /opt/app


# Expose the port that Streamlit will run on
EXPOSE 5000

RUN ls -la .

# Command to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "auditors_opinion_infer_app.py", "--server.port=5000"]

