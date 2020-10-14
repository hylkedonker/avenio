#!/bin/bash

#Execute fragment analysis in batches of three because, for each BAM file, a new index
#is build in memory.
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run1 &
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run1or2 &
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run2 &
wait

python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run3 &
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run4 &
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run5 &
wait

python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run6 &
python3 run_fragment_analysis.py /data/bam/HiSeq_runs/hiseq_run7 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_pilot &
wait

python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run1 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run2 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run3 &
wait

python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run4 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run5 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run6 &
wait

python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run7 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run8 &
python3 run_fragment_analysis.py /data/bam/NextSeq_runs/nextseq_run9 &
