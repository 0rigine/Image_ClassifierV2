#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" Main script of project """

# ---- Imports ----
from NAP import Neural_Agent

# ---- Script ----
if __name__ == '__main__':
    agent = Neural_Agent(
        instance_save_path='google_classifier',
        loops=15,
        data_per_file=2000.,
        data_per_batch=2000.,
        source_path='Resources/Google',
        reuse_data=True,
        dimensions=(299, 299)
    )
    agent.display_prediction('')
