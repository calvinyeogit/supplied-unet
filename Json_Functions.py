#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:21:00 2025

@author: calvin
"""

class Json_Functions:
    def __init__(self):
        pass

    def load_json(self, json_filepath):
        import json
        with open(json_filepath, 'r') as f:
            return json.load(f)

    def save_json(self, data, json_filepath):
        import json
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=4)