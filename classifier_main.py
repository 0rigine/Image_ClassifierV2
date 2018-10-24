#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" Main script of project """

# ---- Imports ----
from NAP import Neural_Agent

# ---- Script ----
if __name__ == '__main__':
    agent = Neural_Agent(loops=2,
                         data_per_file=150.,
                         data_per_batch=70.,
                         source_path='Resources',
                         reuse_data=True,
                         dimensions=(50, 50))
    agent.display_prediction('')
