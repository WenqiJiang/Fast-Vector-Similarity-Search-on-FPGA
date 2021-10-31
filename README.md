# Fast-Vector-Similarity-Search-on-FPGA

under submission at VLDB 2022

# Folder Structure

## FPGA-ANNS-with_network

The ANNS accelerator with TCP/IP network integration. 

## FPGA-ANNS-local

The ANNS accelerator without network kernel.

## bitstreams

Building bitstreams for each accelerator takes >10 hours, we thus provide the pre-built bitstreams (both the network and local versions).

## Faiss_experiments

Baseline experiments using Faiss.

## plots

The plotting scripts and the raw performance results we use in the paper.
