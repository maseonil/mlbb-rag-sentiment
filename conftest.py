import os
import sys

# Tambahkan root directory ke sys.path agar Pytest bisa mengenali folder 'src'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))