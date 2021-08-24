FROM python:3.9 

RUN mkdir diabetes_model

WORKDIR /diabetes_model

COPY ./requirements.txt ./requirements.txt

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt