FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update && \
    apt-get install -qq -y --no-install-recommends \
    python3-pip \
    python3-pysam \
    samtools \
    bcftools \
    bedtools

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Copy the code.
RUN mkdir /package/
RUN mkdir /package/log/ /package/output/
RUN mkdir -p /data/bam/
RUN mkdir -p /metadata/
WORKDIR /package/

#COPY *.ipynb /package/
#COPY *.py /package/
## Copy the genomic and clinical data.
COPY variant_list_20200730.xlsx /metadata/
COPY clinical_20200420.sav /data/

# # Fetch KEGG pathways.
# RUN python3 fetch_path_ways.py
# # Make train and test splits.
# RUN runipy process_final_results.ipynb

# RUN bash /package/fragment_count/analysis_bams.sh