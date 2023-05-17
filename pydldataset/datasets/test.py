import os
import pathlib 
from PIL import Image

class Test:
  def __init__(self): 
    # self.root = os.path.expanduser(root)
    self.root = os.path.join(pathlib.Path(__file__).parent.resolve(), 'tmp/test_data') 
    self.idx = 0
  
  def __len__(self):
    return 6
  
  def __getitem__(self, idx):
    filename = os.path.join(self.root, 'test_' + str(idx) + '.png')
    return Image.open(filename)  

def init():
  return Test()