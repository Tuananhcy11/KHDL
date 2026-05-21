import sys
import os

# Add root folder to sys.path so we can import server.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import GoldDBHandler as handler
