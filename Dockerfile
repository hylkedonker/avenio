FROM ubuntu:20.10
ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update && \
    apt-get install -qq -y --no-install-recommends \
    python3-pip \
    python3-pysam \
    samtools \
    bcftools \
    bedtools \
    git

# Install binary packages so that not all pip packages have to be rebuilt.
RUN apt update -qq && \
    apt install -y -qq \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-sklearn \
    python3-mypy \
    python3-seaborn \
    python3-cvxopt \
    python3-joblib

# Instal libraries for compiling from source.
RUN apt update -qq && \
    apt install -y -qq \
    libblas3 \
    liblapack3 \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libatlas-base-dev \
    cmake \
    g++

RUN git clone https://gitlab.com/hylkedonker/harmonium-models.git /opt/harmonium-models/
RUN pip3 install scikit-survival
RUN pip3 install /opt/harmonium-models/

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
# Copy the code.
RUN mkdir -p /package/models/ /package/log/ /package/output/
RUN mkdir -p /data/bam/
RUN mkdir -p /metadata/
WORKDIR /package/

#COPY *.ipynb /package/
COPY *.py /package/
COPY clinical_20200420.sav gene_annotation.xlsx variant_list_20200730.xlsx /package/
## Copy the genomic and clinical data.
# COPY variant_list_20200730.xlsx /metadata/
# COPY clinical_20200420.sav /data/

# # Fetch KEGG pathways.
# RUN python3 fetch_path_ways.py
# # Make train and test splits.
# RUN runipy process_final_results.ipynb

# RUN bash /package/fragment_count/analysis_bams.sh