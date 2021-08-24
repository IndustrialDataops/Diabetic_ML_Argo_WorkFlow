FROM python:3.9

RUN mkdir streamlit_app

WORKDIR /streamlit_app

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

