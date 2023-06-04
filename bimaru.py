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
#import time
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

class Board:
   """Representação interna de um tabuleiro de Bimaru."""

   def __init__(self, board: np.ndarray, hints: np.ndarray):
      """Construtor da classe"""
      self.board = board
      self.hints = hints
      self.num_boats = [-1, -1, -1]

   def get_value(self, dim: int ,row: int, col: int) -> int:
      """Devolve o valor na respetiva posição do tabuleiro."""
      return self.board[dim][row][col]
   
   def set_value(self, dim: int, row: int, col: int, value: str):
      """Atribui o valor 'value' à posição especificada pelos argumentos 'row' e 'col'."""
      self.board[dim][row][col] = value

   def get_empty_spaces(self):
      """Devolve se existe algum espaço vazio no tabuleiro."""
      return np.any(self.board[1] == 1)
   
   def place_water_diagonals(self, row: int, col: int):
      """Coloca agua nas diagonais de um barco"""

      if row == 0:
         if col == 0:
            self.set_value(1, row + 1, col + 1, WATER)
         elif col == 9:
            self.set_value(1, row + 1, col - 1, WATER)
         else:
            self.set_value(1, row + 1, col - 1, WATER)
            self.set_value(1, row + 1, col + 1, WATER)
      elif row == 9:
         if col == 0:
            self.set_value(1, row - 1, col + 1, WATER)
         elif col == 9:
            self.set_value(1, row - 1, col - 1, WATER)
         else:
            self.set_value(1, row - 1, col - 1, WATER)
            self.set_value(1, row - 1, col + 1, WATER)
      elif col == 0:
         self.set_value(1, row - 1, col + 1, WATER)
         self.set_value(1, row + 1, col + 1, WATER)
      elif col == 9:
         self.set_value(1, row - 1, col - 1, WATER)
         self.set_value(1, row + 1, col - 1, WATER)
      else:
         self.set_value(1, row - 1, col - 1, WATER)
         self.set_value(1, row - 1, col + 1, WATER)
         self.set_value(1, row + 1, col - 1, WATER)
         self.set_value(1, row + 1, col + 1, WATER)

   def place_boat(self, row: int, col: int):
      """Coloca um barco na posição especificada pelos argumentos 'row' e 'col'."""
      self.board[0][row][col] = BOAT
      self.board[1][row][col] = WATER
      self.place_water_diagonals(row, col)

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
         if (i < 9):
            line += '\n'
         
      print(line)

   def check_board_validity(self) ->bool:
      """Verifica se o board é válido"""

      #sums the rows and columns of the first two matrices
      empy_n_boat_row_sums = np.sum(self.board[:2], axis = 2)
      empy_n_boat_col_sums = np.sum(self.board[:2], axis = 1)

      rows_boats_minus_hints  = empy_n_boat_row_sums[0] - self.hints[0]
      if(np.any(rows_boats_minus_hints > 0)): return False
      rows_boats_plus_emptys_minus_hints =  rows_boats_minus_hints + empy_n_boat_row_sums[1]
      if(np.any(rows_boats_plus_emptys_minus_hints < 0)): return False

      cols_boats_minus_hints = empy_n_boat_col_sums[0] - self.hints[1]
      if(np.any(cols_boats_minus_hints > 0)): return False
      cols_boats_plus_emptys_minus_hints = cols_boats_minus_hints + empy_n_boat_col_sums[1]
      if(np.any(cols_boats_plus_emptys_minus_hints < 0)): return False

      return True

   def check_correct_boats(self) ->bool:
      """Verifica se os barcos estão corretos"""
      boats = self.board[0]

      boats_colums_sum_1 = boats[:-1, :] + boats[1:, :]
      boats_rows_sum_1 = boats[:, :-1] + boats[:, 1:]
      contratorpecidos_spaces = np.count_nonzero(boats_colums_sum_1 == 2) + np.count_nonzero(boats_rows_sum_1 == 2)

      boats_colums_sum_2 = boats[2:, :] + boats_colums_sum_1[:-1, :]
      boats_rows_sum_2 = boats[:, 2:] + boats_rows_sum_1[:, :-1]
      cruzador_spaces = np.count_nonzero(boats_colums_sum_2 == 3) + np.count_nonzero(boats_rows_sum_2 == 3)

      boats_colums_sum_3 = boats[3:, :] + boats_colums_sum_2[:-1, :]
      boats_rows_sum_3 = boats[:, 3:] + boats_rows_sum_2[:, :-1]
      couracado_spaces = np.count_nonzero(boats_colums_sum_3 == 4) + np.count_nonzero(boats_rows_sum_3 == 4)

      contratorpecidos_count = contratorpecidos_spaces - 2*cruzador_spaces + couracado_spaces
      cruzador_count = cruzador_spaces - 2*couracado_spaces

      self.num_boats = [couracado_spaces, cruzador_count, contratorpecidos_count]

      return contratorpecidos_count == N_CONTRATORPECIDOS and cruzador_count == N_CRUZADOR and couracado_spaces == N_COURACADO

   def get_boats_to_place(self) ->bool:
      '''Vê qual é o maior barco que falta colocar e os possiveis lugares onde ele pode ser colocado'''
      boats = self.board[0]
      boats_colums_sum_1 = boats[:-1, :] + boats[1:, :]
      boats_rows_sum_1 = boats[:, :-1] + boats[:, 1:]
      boats_colums_sum_2 = boats[2:, :] + boats_colums_sum_1[:-1, :]
      boats_rows_sum_2 = boats[:, 2:] + boats_rows_sum_1[:, :-1]
      boats_colums_sum_3 = boats[3:, :] + boats_colums_sum_2[:-1, :]
      boats_rows_sum_3 = boats[:, 3:] + boats_rows_sum_2[:, :-1]

      emptys = self.board[1] 
      emptys_colums_sum_1 = emptys[:-1, :] + emptys[1:, :]
      emptys_rows_sum_1 = emptys[:, :-1] + emptys[:, 1:]
      emptys_colums_sum_2 = emptys[2:, :] + emptys_colums_sum_1[:-1, :]
      emptys_rows_sum_2 = emptys[:, 2:] + emptys_rows_sum_1[:, :-1]
      emptys_colums_sum_3 = emptys[3:, :] + emptys_colums_sum_2[:-1, :]
      emptys_rows_sum_3 = emptys[:, 3:] + emptys_rows_sum_2[:, :-1]

      if(self.num_boats[0] < N_COURACADO):
         indices_colums = np.where(np.logical_and(boats_colums_sum_3 + emptys_colums_sum_3 == 4, boats_colums_sum_3 < 4 ))
         indices_rows = np.where(np.logical_and(boats_rows_sum_3 + emptys_rows_sum_3 == 4, boats_rows_sum_3 < 4))
         return (4, indices_rows, indices_colums)

      if(self.num_boats[1] < N_CRUZADOR):
         indices_colums = np.where(np.logical_and(boats_colums_sum_2 + emptys_colums_sum_2 == 3, boats_colums_sum_2 < 3))
         indices_rows = np.where(np.logical_and(boats_rows_sum_2 + emptys_rows_sum_2 == 3, boats_rows_sum_2 < 3))
         return (3, indices_rows, indices_colums)
      
      if(self.num_boats[2] < N_CONTRATORPECIDOS):
         indices_colums = np.where(np.logical_and(boats_colums_sum_1 + emptys_colums_sum_1 == 2, boats_colums_sum_1 < 2))
         indices_rows = np.where(np.logical_and(boats_rows_sum_1 + emptys_rows_sum_1 == 2, boats_rows_sum_1 < 2))
         return (2, indices_rows, indices_colums)
      
      return (-1, None, None)

   def fill_water_boats(self) -> bool:
      """Preenche as posições que só podem ter água ou barco."""
      diff = True
      while diff:
         #sums the rows and columns of the first two matrices
         empy_n_boat_row_sums = np.sum(self.board, axis = 2)
         empy_n_boat_col_sums = np.sum(self.board, axis = 1)

         #subtracts the sum of the rows and columns from the third matrix
         empty_n_boat_rows_diff = np.subtract(empy_n_boat_row_sums[0], self.hints[0])
         rows_diff = np.subtract(np.add(empy_n_boat_row_sums[0], empy_n_boat_row_sums[1]), self.hints[0])
         empty_n_boat_cols_diff = np.subtract(empy_n_boat_col_sums[0], self.hints[1])
         cols_diff = np.subtract(np.add(empy_n_boat_col_sums[0], empy_n_boat_col_sums[1]), self.hints[1])

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
            cols_equal_to_one_indices = np.where(self.board[1, k, :] == BOAT)[0]
            for m in cols_equal_to_one_indices:
               self.place_boat(k, m)
            self.board[1, k, :] = np.zeros(self.board.shape[2])

         #puts water in rows
         for i in rows_indices_1:
            self.board[1, i, :] = np.zeros(self.board.shape[2])
         
         #puts boats in columns
         for l in cols_indices_2:
            rows_equal_to_one_indices = np.where(self.board[1, :, l] == BOAT)[0]
            for n in rows_equal_to_one_indices:
               self.place_boat(n, l)
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
      rows = [int(x) for x in row_line[1:]]

      #column with hint
      column_line = stdin.readline().rstrip('\n').split('\t')
      columns = [int(x) for x in column_line[1:]]
      num_hints = int(input())

      #creating board with letters
      original_board = np.zeros((10, 10), dtype = str)

      #creating board with water and board with boats
      board_boats = np.zeros((10, 10), dtype = int)
      board_water = np.ones((10, 10), dtype = int)
      board_pad_boats = np.pad(board_boats, 1)
      board_pad_water = np.pad(board_water, 1)

      #adding hints to board
      for i in range(num_hints):
         hint_line = stdin.readline().rstrip('\n').split('\t')
         hint = hint_line[3]
         hint_row = int(hint_line[1])
         hint_col = int(hint_line[2])
         original_board[hint_row][hint_col] = hint
         hint_row += 1
         hint_col += 1
         if hint[0] == 'W':
            board_pad_water[hint_row][hint_col] = WATER
         else:
            board_pad_boats[hint_row][hint_col] = BOAT
            board_pad_water[hint_row][hint_col] = WATER
            if hint[0] == 'T':
               board_pad_boats[hint_row + 1][hint_col] = BOAT
               board_pad_water[hint_row + 1][hint_col] = WATER
               board_pad_water[hint_row - 1][hint_col] = WATER
               board_pad_water[hint_row - 1][hint_col + 1] = WATER
               board_pad_water[hint_row][hint_col + 1] = WATER
               board_pad_water[hint_row + 1][hint_col + 1] = WATER
               board_pad_water[hint_row + 2][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col - 1] = WATER
               board_pad_water[hint_row][hint_col - 1] = WATER
               board_pad_water[hint_row + 1][hint_col - 1] = WATER
               board_pad_water[hint_row + 2][hint_col - 1] = WATER

            elif hint[0] == 'B':
               board_pad_boats[hint_row - 1][hint_col] = BOAT
               board_pad_water[hint_row - 1][hint_col] = WATER
               board_pad_water[hint_row + 1][hint_col] = WATER
               board_pad_water[hint_row + 1][hint_col - 1] = WATER
               board_pad_water[hint_row][hint_col - 1] = WATER
               board_pad_water[hint_row - 1][hint_col - 1] = WATER
               board_pad_water[hint_row - 2][hint_col - 1] = WATER
               board_pad_water[hint_row + 1][hint_col + 1] = WATER
               board_pad_water[hint_row][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col + 1] = WATER
               board_pad_water[hint_row - 2][hint_col + 1] = WATER
            
            elif hint[0] == 'R':
               board_pad_boats[hint_row][hint_col - 1] = BOAT
               board_pad_water[hint_row][hint_col - 1] = WATER
               board_pad_water[hint_row][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col] = WATER
               board_pad_water[hint_row - 1][hint_col - 1] = WATER
               board_pad_water[hint_row + 1][hint_col + 1] = WATER
               board_pad_water[hint_row + 1][hint_col] = WATER
               board_pad_water[hint_row + 1][hint_col - 1] = WATER
               board_pad_water[hint_row - 1][hint_col - 2] = WATER
               board_pad_water[hint_row + 1][hint_col - 2] = WATER

            elif hint[0] == 'L':
               board_pad_boats[hint_row][hint_col + 1] = BOAT
               board_pad_water[hint_row][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col - 1] = WATER
               board_pad_water[hint_row - 1][hint_col] = WATER
               board_pad_water[hint_row - 1][hint_col + 1] = WATER
               board_pad_water[hint_row][hint_col - 1] = WATER
               board_pad_water[hint_row + 1][hint_col - 1] = WATER
               board_pad_water[hint_row + 1][hint_col] = WATER
               board_pad_water[hint_row + 1][hint_col + 1] = WATER
               board_pad_water[hint_row - 1][hint_col + 2] = WATER
               board_pad_water[hint_row + 1][hint_col + 2] = WATER

            elif hint[0] == 'C':
               board_pad_water[hint_row - 1][hint_col] = WATER
               board_pad_water[hint_row - 1][hint_col + 1] = WATER
               board_pad_water[hint_row][hint_col + 1] = WATER
               board_pad_water[hint_row + 1][hint_col + 1] = WATER
               board_pad_water[hint_row + 1][hint_col] = WATER
               board_pad_water[hint_row + 1][hint_col - 1] = WATER
               board_pad_water[hint_row][hint_col - 1] = WATER
               board_pad_water[hint_row - 1][hint_col - 1] = WATER
            
            if hint[0] == 'M':
                  if hint_row == 1:
                     board_pad_boats[hint_row][hint_col - 1] = BOAT
                     board_pad_boats[hint_row][hint_col + 1] = BOAT
                     board_pad_water[hint_row][hint_col - 1] = WATER
                     board_pad_water[hint_row][hint_col + 1] = WATER
                     if hint_col == 2:
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                        board_pad_water[hint_row + 1][hint_col + 2] = WATER
                     elif hint_col == 9:
                        board_pad_water[hint_row + 1][hint_col - 2] = WATER
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                     else:
                        board_pad_water[hint_row + 1][hint_col - 2] = WATER
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                        board_pad_water[hint_row + 1][hint_col + 2] = WATER
                  elif hint_row == 10:
                     board_pad_boats[hint_row][hint_col - 1] = BOAT
                     board_pad_boats[hint_row][hint_col + 1] = BOAT
                     board_pad_water[hint_row][hint_col - 1] = WATER
                     board_pad_water[hint_row][hint_col + 1] = WATER
                     if hint_col == 2:
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row - 1][hint_col] = WATER
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                        board_pad_water[hint_row - 1][hint_col + 2] = WATER
                     elif hint_col == 9:
                        board_pad_water[hint_row - 1][hint_col - 2] = WATER
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row - 1][hint_col] = WATER
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                     else:
                        board_pad_water[hint_row - 1][hint_col - 2] = WATER
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row - 1][hint_col] = WATER
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                        board_pad_water[hint_row - 1][hint_col + 2] = WATER
                  elif hint_col == 1:
                     board_pad_boats[hint_row - 1][hint_col] = BOAT
                     board_pad_boats[hint_row + 1][hint_col] = BOAT
                     board_pad_water[hint_row - 1][hint_col] = WATER
                     board_pad_water[hint_row + 1][hint_col] = WATER
                     if hint_row == 2:
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                        board_pad_water[hint_row][hint_col + 1] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                        board_pad_water[hint_row + 2][hint_col + 1] = WATER
                     elif hint_row == 9:
                        board_pad_water[hint_row - 2][hint_col + 1] = WATER
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                        board_pad_water[hint_row][hint_col + 1] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                     else:
                        board_pad_water[hint_row - 2][hint_col + 1] = WATER
                        board_pad_water[hint_row - 1][hint_col + 1] = WATER
                        board_pad_water[hint_row][hint_col + 1] = WATER
                        board_pad_water[hint_row + 1][hint_col + 1] = WATER
                        board_pad_water[hint_row + 2][hint_col + 1] = WATER
                  elif hint_col == 10:
                     board_pad_boats[hint_row - 1][hint_col] = BOAT
                     board_pad_boats[hint_row + 1][hint_col] = BOAT
                     board_pad_water[hint_row - 1][hint_col] = WATER
                     board_pad_water[hint_row + 1][hint_col] = WATER
                     if hint_row == 2:
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                        board_pad_water[hint_row + 2][hint_col - 1] = WATER
                     elif hint_row == 9:
                        board_pad_water[hint_row - 2][hint_col - 1] = WATER
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                     else:
                        board_pad_water[hint_row - 2][hint_col - 1] = WATER
                        board_pad_water[hint_row - 1][hint_col - 1] = WATER
                        board_pad_water[hint_row][hint_col - 1] = WATER
                        board_pad_water[hint_row + 1][hint_col - 1] = WATER
                        board_pad_water[hint_row + 2][hint_col - 1] = WATER
                  else:
                     board_pad_water[hint_row - 1][hint_col - 1] = WATER
                     board_pad_water[hint_row - 1][hint_col + 1] = WATER
                     board_pad_water[hint_row + 1][hint_col - 1] = WATER
                     board_pad_water[hint_row + 1][hint_col + 1] = WATER
      
      #creating board with numbers
      board = np.ones((2,10,10), dtype=int)
      board[0] = board_pad_boats[1:11, 1:11]
      board[1] = board_pad_water[1:11, 1:11]
      
      #creating vector with hints
      hints = np.zeros((2,10), dtype=int)

      #adding hints to vector
      hints[0][:] = rows
      hints[1][:] = columns

      return Board(original_board, hints), Board(board, hints)

class Bimaru(Problem):
   def __init__(self, board: Board):
      """O construtor especifica o estado inicial."""
      state = BimaruState(board)
      super().__init__(state)

   def actions(self, state: BimaruState):
      """Retorna uma lista de ações que podem ser executadas a
      partir do estado passado como argumento."""
      actions = []

      if(not state.board.get_empty_spaces()) or (not state.board.check_board_validity()):
         return actions

      (size, row_boats, col_boats) = state.board.get_boats_to_place()
      if(size == -1):
            return [(-1, WATER, 0, 0), (-1, BOAT, 0, 0)]
         
      for i in range(0, len(row_boats[0])):
         actions.append((size, 0, row_boats[0][i], row_boats[1][i]))

      for j in range(0, len(col_boats[0])):
         actions.append((size, 1, col_boats[0][j], col_boats[1][j]))
      return actions
         
   def result(self, state: BimaruState, action):
      """Retorna o estado resultante de executar a 'action' sobre
      'state' passado como argumento. A ação a executar deve ser uma
      das presentes na lista obtida pela execução de
      self.actions(state)."""

      new_board = Board(np.copy(state.board.board), state.board.hints)
      new_state = BimaruState(new_board)
      if action[0] == -1:
         indices = np.where(new_state.board.board[1] == 1)
         if action[1] == BOAT:
            new_state.board.place_boat(indices[0][0], indices[1][0])
         new_state.board.board[0][indices[0][0]][indices[1][0]] = action[1]
         new_state.board.board[1][indices[0][0]][indices[1][0]] = WATER
      elif action[1] == 0:
         for i in range(action[0]):
            new_state.board.place_boat(action[2], action[3] + i)
      elif action[1] == 1:
         for i in range(action[0]):
            new_state.board.place_boat(action[2] + i, action[3])

      new_state.board.fill_water_boats()

      return new_state

   def goal_test(self, state: BimaruState):
      """Retorna True se e só se o estado passado como argumento é
      um estado objetivo. Deve verificar se todas as posições do tabuleiro
      estão preenchidas de acordo com as regras do problema."""
      if not state.board.check_correct_boats(): return False
      if state.board.get_empty_spaces(): return False
      if not state.board.check_board_validity(): return False
      return True

   def h(self, node: Node):
      """Função heuristica utilizada para a procura A*."""
      # TODO
      pass

if __name__ == "__main__":
   original_board, board = Board.parse_instance()
   board.fill_water_boats()
   board_state = BimaruState(board)
   bimaru = Bimaru(board)
   solution = depth_first_tree_search(bimaru)
   solution.state.board.print_board(original_board.board)
   