# Dockerfile for the project

# pull miniconda image
FROM continuumio/miniconda3

# copy local files into container
COPY requirements.txt asail_streamlit.py /tmp/
COPY models /tmp/models
COPY utils /tmp/utils
COPY data /tmp/data
COPY checkpoints /tmp/checkpoints


# install python 3.8.3
RUN conda install python=3.8.3
ENV PORT 8501

# change directory
WORKDIR /tmp

# install dependencies
RUN apt-get update && apt-get install -y vim g++
RUN pip install cython numpy scipy  pandas
RUN pip install -r requirements.txt

# run commands
CMD ["streamlit", "run", "asail_streamlit.py", "--", "--load_epoch", "amazon_3", "--gpu_ids", "-1"]
