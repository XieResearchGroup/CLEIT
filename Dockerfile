FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN apt-get update && apt-get install -y build-essential
RUN conda install pandas numpy scikit-learn pyarrow && pip install xgboost


