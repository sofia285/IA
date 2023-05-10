import sys
from sys import stdin
#import numpy as np
class read:
   def read_input(self):
      # Read the input from standard input and split it into a list
      row = stdin.readline().rstrip('\n').split('\t')
      columns = stdin.readline().rstrip('\n').split('\t')
      hints = stdin.readline().rstrip('\n').split('\t')
      print(row)
      print(columns)
      print(hints)
      i = int(hints[0])
      while i > 0:
         hint = stdin.readline().rstrip('\n').split('\t')
         i = i - 1
         print(hint)

class OtherClass:
    def call_my_function(self):
        my_instance = read()
        my_instance.read_input()


if __name__ == '__main__':
   other_instance = OtherClass()
   other_instance.call_my_function()

