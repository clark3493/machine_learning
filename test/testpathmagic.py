import os
import sys


test_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(test_dir)
src_dir = os.path.join(project_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)