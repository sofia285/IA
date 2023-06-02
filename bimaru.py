# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 72:
# 102835 Sofia Paiva
# 102904 Mariana Miranda

WATER = 0
BOAT = 1
EMPTY = 2

CIRCLE = 'c'
TOP = 't'
MIDDLE = 'm'
BOTTOM = 'b'
LEFT = 'l'
RIGHT = 'r'
N_COURACADO = 1
N_CRUZADOR = 2
N_CONTRATORPECIDOS = 3
N_SUBMARINO = 4

import numpy as np
import sys
from sys import stdin

from search import (
   Problem,
   Node,
   astar_search,
   breadth_first_tree_search,
   depth_first_tree_search,
   greedy_search,
   recursive_best_first_search,
)


class BimaruState:
   state_id = 0
   

   def __init__(self, board):
      self.board = board
      self.id = BimaruState.state_id
      BimaruState.state_id += 1

   def __lt__(self, other):
      return self.id < other.id

   def set_num_boats(self, boat_type, num):
      self.num_boats[boat_type] = num

   def get_num_boats(self, boat_type):
      return self.num_boats[boat_type]
   
   # TODO: outros metodos da classe


class Board:
   """Representação interna de um tabuleiro de Bimaru."""

   def __init__(self, board: np.ndarray):
      """Construtor da classe"""
      self.board = board
      self.num_boats ={
         'couracado': 0,
         'cruzador': 0,
         'contratorpecidos': 0,
         'submarino': 0,
      }


   def get_value(self, dim: int ,row: int, col: int) -> int:
      """Devolve o valor na respetiva posição do tabuleiro."""

      return self.board[dim][row][col]
   
   def set_value(self, dim: int, row: int, col: int, value: str):
      """Atribui o valor 'value' à posição especificada pelos argumentos 'row' e 'col'."""
      self.board[dim][row][col] = value

   def adjacent_vertical_values(self, dim: int, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente acima e abaixo,
      respectivamente."""
      if row == 0:
         return None, self.get_value(dim, row + 1, col)
      elif row == 9:
         return self.get_value(dim, row - 1, col), None
      else:
         return self.get_value(dim, row - 1, col), self.get_value(dim, row + 1, col)

   def adjacent_horizontal_values(self, dim: int, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente à esquerda e à direita,
      respectivamente."""
      if col == 0:
         return None, self.get_value(dim, row, col + 1)
      elif col == 9:
         return self.get_value(dim, row, col - 1), None
      else:
         return self.get_value(dim, row, col - 1), self.get_value(dim, row, col + 1)
      
   def adjacent_diagonal_top_values(self, dim: int, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente nas diagonais superiores"""
      if row == 0: 
         return None, None
      elif col == 0:
         return None, self.get_value(dim, row - 1, col + 1)
      elif col == 9:
            return self.get_value(dim, row - 1, col - 1), None
      else:
         return self.get_value(dim, row - 1, col - 1), self.get_value(dim, row - 1, col + 1)
   
   def adjacent_diagonal_bottom_values(self, dim: int, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente nas diagonais inferiores"""
      if row == 9: 
         return None, None
      elif col == 0:
         return None, self.get_value(dim, row + 1, col + 1)
      elif col == 9:
            return self.get_value(dim, row + 1, col - 1), None
      else:
         return self.get_value(dim, row + 1, col - 1), self.get_value(dim, row + 1, col + 1)

   def get_empty_spaces_count(self):
      """Devolve se existe algum espaço vazio no tabuleiro."""
      return np.any(self.board[1] == 1)

   def has_been_visited(self, dim: int, row: int, col: int) -> bool:
      """Devolve se a posição já foi visitada."""
      return self.get_value(dim, row, col) == 1

   def print_board(self, original_board):
      """Imprime o tabuleiro."""
      boats_pad = np.pad(self.board[0], 1)
      boats_adjacents = boats_pad[0:10, 1:11] + 2 * boats_pad[1:11, 1:11] + 4 * boats_pad[2:12, 1:11] + 8 * boats_pad[1:11, 0:10] + 16 * boats_pad[1:11, 2:12]

      line = ''
      for i in range (10):
         for j in range (10):
            if original_board[i][j] != '':
               line+= original_board[i][j];
               continue;
            if boats_adjacents[i][j] == 2:
               line += CIRCLE
            elif boats_adjacents[i][j] == 3:
               line += BOTTOM
            elif boats_adjacents[i][j] == 6:
               line += TOP
            elif boats_adjacents[i][j] == 10:
               line += RIGHT
            elif boats_adjacents[i][j] == 18:
               line += LEFT
            elif boats_adjacents[i][j] == 7 or boats_adjacents[i][j] == 26:
               line += MIDDLE
            else:
               line += '.'
         line += '\n'
         
      print(line)
   
   def get_empty_spaces_count(self):
      """Devolve se existe algum espaço vazio no tabuleiro."""
      return np.any(board[1] == 1)

   def fill_water_boats(self) ->bool:
      """Preenche as posições que só podem ter água ou barcos."""
      diff = True
      while diff:
         #sums the rows and columns of the first two matrices
         empy_n_boat_row_sums = np.sum(self.board[:2], axis = 2)
         empy_n_boat_col_sums = np.sum(self.board[:2], axis = 1)

         #subtracts the sum of the rows and columns from the third matrix
         empty_n_boat_rows_diff = np.subtract(empy_n_boat_row_sums[0], self.board[2, :, 9])
         rows_diff = np.subtract(np.add(empy_n_boat_row_sums[0], empy_n_boat_row_sums[1]), self.board[2, :, 9])
         empty_n_boat_cols_diff = np.subtract(empy_n_boat_col_sums[0], self.board[2, :, 0])
         cols_diff = np.subtract(np.add(empy_n_boat_col_sums[0], empy_n_boat_col_sums[1]), self.board[2, :, 0])

         #finds the indices of the rows that are equal to zero
         rows_equal_to_zero_indices_1 = np.where(empty_n_boat_rows_diff == 0)[0]
         rows_equal_to_zero_indices_2 = np.where(rows_diff == 0)[0]
         rows_zeros_indices = np.where(np.all(self.board[1] == 0, axis = 1))[0]
         rows_indices_1 = np.setdiff1d(rows_equal_to_zero_indices_1, rows_zeros_indices)
         rows_indices_2 = np.setdiff1d(rows_equal_to_zero_indices_2, rows_zeros_indices)

         #finds the indices of the columns that are equal to zero
         cols_equal_to_zero_indices_1 = np.where(empty_n_boat_cols_diff == 0)[0]
         cols_equal_to_zero_indices_2 = np.where(cols_diff == 0)[0]
         cols_zeros_indices = np.where(np.all(self.board[1] == 0, axis = 0))[0]
         cols_indices_1 = np.setdiff1d(cols_equal_to_zero_indices_1, cols_zeros_indices)
         cols_indices_2 = np.setdiff1d(cols_equal_to_zero_indices_2, cols_zeros_indices)

         #checks if there are any differences
         diff = np.size(rows_indices_2) != 0 or np.size(rows_indices_1) != 0 or np.size(cols_indices_2) != 0 or np.size(cols_indices_1) != 0
         
         #puts boats in rows
         for k in rows_indices_2:
            self.board[0, k, :] = np.add(self.board[1, k, :], self.board[0, k, :])
            self.board[1, k, :] = np.zeros(self.board.shape[2])

         #puts water in rows
         for i in rows_indices_1:
            self.board[1, i, :] = np.zeros(self.board.shape[2])
         
         #puts boats in columns
         for l in cols_indices_2:
            self.board[0, :, l] = np.add(self.board[1, :, l], self.board[0, :, l])
            self.board[1, :, l] = np.zeros(self.board.shape[1])
      
         #puts water in columns
         for j in cols_indices_1:
            self.board[1, :, j] = np.zeros(self.board.shape[1])

   @staticmethod
   def parse_instance():
      """Lê o test do standard input (stdin) que é passado como argumento
      e retorna uma instância da classe Board."""

      #row with hint
      row_line = stdin.readline().rstrip('\n').split('\t')
      row = [int(x) for x in row_line[1:]]

      #column with hint
      column_line = stdin.readline().rstrip('\n').split('\t')
      column = [int(x) for x in column_line[1:]]
      num_hints = int(input())

      #creating board with letters
      original_board = np.zeros((11, 11), dtype=str)
      original_board[:-1, -1] = row
      original_board[-1, :-1] = column

      #adding hints to board
      for i in range(num_hints):
         hint_line = stdin.readline().rstrip('\n').split('\t')
         hint = hint_line[3]
         hint_row = int(hint_line[1])
         hint_column = int(hint_line[2])
         original_board[hint_row][hint_column] = hint
      
      #creating board with numbers
      board = np.ones((3,10,10), dtype=int)
      board[0] = np.zeros((10, 10), dtype=int)

      #adding hints to board
      for i in range (10):
         board[2][i][9] = int(original_board[i][10])
         board[2][i][0] = int(original_board[10][i])
      
      #adding boats and water to new board
      for row in range (10):
         for col in range (10):
            if original_board[row][col] == 'W':
               board[1][row][col] = WATER
            elif original_board[row][col] != '':
               board[0][row][col] = BOAT
               board[1][row][col] = WATER
               if original_board[row][col] == 'M':
                  if row == 0:
                     board[1][row][col - 1] = BOAT
                     board[1][row][col + 1] = BOAT
                     board[1][row][col - 1] = WATER
                     board[1][row][col + 1] = WATER
                     if col == 1:
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col + 2] = WATER
                     elif col == 8:
                        board[1][row + 1][col - 2] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                     else:
                        board[1][row + 1][col - 2] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col + 2] = WATER
                  elif row == 9:
                     board[1][row][col - 1] = BOAT
                     board[1][row][col + 1] = BOAT
                     board[1][row][col - 1] = WATER
                     board[1][row][col + 1] = WATER
                     if col == 1:
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col + 2] = WATER
                     elif col == 8:
                        board[1][row - 1][col - 2] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                     else:
                        board[1][row - 1][col - 2] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col + 2] = WATER
                  elif col == 0:
                     board[1][row - 1][col] = BOAT
                     board[1][row + 1][col] = BOAT
                     board[1][row - 1][col] = WATER
                     board[1][row + 1][col] = WATER
                     if row == 1:
                        board[1][row - 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 2][col + 1] = WATER
                     elif row == 8:
                        board[1][row - 2][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                     else:
                        board[1][row - 2][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 2][col + 1] = WATER
                  elif col == 9:
                     board[1][row - 1][col] = BOAT
                     board[1][row + 1][col] = BOAT
                     board[1][row - 1][col] = WATER
                     board[1][row + 1][col] = WATER
                     if row == 1:
                        board[1][row - 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 2][col - 1] = WATER
                     elif row == 8:
                        board[1][row - 2][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                     else:
                        board[1][row - 2][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 2][col - 1] = WATER
                  else:
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 1][col + 1] = WATER

               elif original_board[i][col] == 'T':
                  board[0][row + 1][col] = BOAT
                  board[1][row + 1][col] = WATER
                  if row == 0:
                     if col == 0:
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 2][col + 1] = WATER
                     elif col == 9:
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 2][col - 1] = WATER
                     else:
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 2][col + 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 2][col - 1] = WATER
                  elif row == 8:
                     if col == 0:
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                     elif col == 9:
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                     else:
                        board[1][row - 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                  elif col == 0:
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row + 2][col + 1] = WATER
                  elif col == 9:
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 2][col - 1] = WATER
                  else:
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row + 2][col + 1] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 2][col - 1] = WATER

               elif original_board[row][col] == 'B':
                  board[0][row - 1][col] = BOAT
                  board[1][row - 1][col] = WATER
                  if row == 1:
                     if col == 0:
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                     elif col == 9:
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                     else:
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                  elif row == 9:
                     if col == 0:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 2][col + 1] = WATER
                     elif col == 9:
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 2][col - 1] = WATER
                     else:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 2][col + 1] = WATER
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 2][col - 1] = WATER
                  elif col == 0:
                     board[1][row][col + 1] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 2][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col + 1] = WATER
                  elif col == 9:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 2][col - 1] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 2][col - 1] = WATER
                     board[1][row][col + 1] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 2][col + 1] = WATER
                     board[1][row + 1][col + 1] = WATER
   
               elif original_board[row][col] == 'R':
                  board[0][row][col - 1] = BOAT
                  board[1][row][col - 1] = WATER
                  if row == 0:
                     if col == 1:
                        board[1][row][col + 1] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col - 1] = WATER
                     elif col == 9:
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col - 2] = WATER
                        board[1][row][col - 2] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col + 2] = WATER
                  elif row == 0:
                     if col == 1:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                     elif col == 9:
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col - 2] = WATER
                     else:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row + 1][col - 2] = WATER
                  elif col == 1:
                     board[1][row][col + 1] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col - 1] = WATER
                  elif col == 9:
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 1][col - 2] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 1][col - 2] = WATER
                  else:
                     board[1][row][col + 1] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row - 1][col - 2] = WATER
                     board[1][row + 1][col - 2] = WATER

               elif original_board[row][col] == 'L':
                  board[0][row][col + 1] = BOAT
                  board[1][row][col + 1] = WATER
                  if row == 0:
                     if col == 0:
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col + 2] = WATER
                     elif col == 8:
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row + 1][col - 1] = WATER
                        board[1][row + 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                        board[1][row + 1][col + 2] = WATER
                  elif row == 9:
                     if col == 0:
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col + 2] = WATER
                     elif col == 8:
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                        board[1][row - 1][col + 2] = WATER
                  elif col == 0:
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 1][col + 2] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row + 1][col + 2] = WATER
                  elif col == 8:
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col + 1] = WATER
                  else:
                     board[1][row - 1][col - 1] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row + 1][col + 1] = WATER
                     board[1][row - 1][col + 2] = WATER
                     board[1][row + 1][col + 2] = WATER
                  
               elif original_board[row][col] == 'C':
                  if row == 0:
                     if col == 0:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                     elif col == 9:
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                  elif row == 9:
                     if col == 0:
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                     elif col == 9:
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                  elif col == 0:
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                  elif col == 9:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row][col + 1] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row - 1][col - 1] = WATER
                     board[1][row + 1][col - 1] = WATER
                     board[1][row - 1][col + 1] = WATER

      return Board(original_board), Board(board)


class Bimaru(Problem):
   def __init__(self, board: Board):
      """O construtor especifica o estado inicial."""
      state = BimaruState(board)
      super().__init__(state)

   def actions(self, state: BimaruState):
      """Retorna uma lista de ações que podem ser executadas a
      partir do estado passado como argumento."""
      # TODO
      pass

   def result(self, state: BimaruState, action):
      """Retorna o estado resultante de executar a 'action' sobre
      'state' passado como argumento. A ação a executar deve ser uma
      das presentes na lista obtida pela execução de
      self.actions(state)."""
      # TODO
      pass

   def goal_test(self, state: BimaruState):
      """Retorna True se e só se o estado passado como argumento é
      um estado objetivo. Deve verificar se todas as posições do tabuleiro
      estão preenchidas de acordo com as regras do problema."""
      return state.board.get_empty_spaces_count()

   def h(self, node: Node):
      """Função heuristica utilizada para a procura A*."""
      board = node.state.board
      return board.count_pos_with_two_actions


if __name__ == "__main__":

   # Ler o ficheiro do standard input,
   original_board, board = Board.parse_instance()
   # Usar uma técnica de procura para resolver a instância,
   board.fill_water_boats()
   board_state = BimaruState(board)
   #print(board.board)
   # Retirar a solução a partir do nó resultante,
   # Imprimir para o standard output no formato indicado.
   board.print_board(original_board.board)
   