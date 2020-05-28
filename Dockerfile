FROM jupyter/datascience-notebook:1386e2046833
ENV DEBIAN_FRONTEND=noninteractive

USER root

COPY requirements.txt /tmp
# Copy the code.
COPY *.ipynb .
COPY *.py .
# Copy the genomic and clinical data.
COPY variant_list_20200409.xlsx .
COPY clinical_20200420.sav .
RUN pip install -r /tmp/requirements.txt

# Fetch KEGG pathways.
RUN python3 fetch_path_ways.py
# Make train and test splits.
RUN runipy process_final_results.ipynb
