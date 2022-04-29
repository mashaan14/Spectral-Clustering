# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:52:41 2020

@author: mals6571
"""
import os
from core.util import get_project_root

ProjectDir = get_project_root()
for r in range(10):
    exec(open(os.path.join(ProjectDir,'applications/run.py')).read())