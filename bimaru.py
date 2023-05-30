# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 72:
# 102835 Sofia Paiva
# 102904 Mariana Miranda

WATER = 0
BOAT = 1
EMPTY = 1

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
      self.num_boats ={
         'couracado': 0,
         'cruzador': 0,
         'contratorpecidos': 0,
         'submarino': 0,
      }

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


   def get_value(self, row: int, col: int) -> str:
      """Devolve o valor na respetiva posição do tabuleiro."""

      return self.board[row][col]

   def adjacent_vertical_values(self, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente acima e abaixo,
      respectivamente."""
      if row == 0:
         return None, self.get_value(row + 1, col)
      elif row == 9:
         return self.get_value(row - 1, col), None
      else:
         return self.get_value(row - 1, col), self.get_value(row + 1, col)

   def adjacent_horizontal_values(self, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente à esquerda e à direita,
      respectivamente."""
      if col == 0:
         return None, self.get_value(row, col + 1)
      elif col == 9:
         return self.get_value(row, col - 1), None
      else:
         return self.get_value(row, col - 1), self.get_value(row, col + 1)
      

   def get_row(self, row: int) -> np.ndarray:
      """Devolve a linha especificada pelo argumento 'row'."""
      return self.board[row]
   
   def get_column(self, col: int) -> np.ndarray:
      """Devolve a coluna especificada pelo argumento 'col'."""
      return self.board[:, col]
      
   def set_value(self, row: int, col: int, value: str):
      """Atribui o valor 'value' à posição especificada pelos
      argumentos 'row' e 'col'."""
      self.board[row][col] = value

   def get_col_count(self, col: int) -> int:
     """Deveolve a contagem da coluna especificada pelo argumento 'col'."""
     return int(self.board[10][col])
   
   def get_row_count(self, row: int) -> int:
       """Deveolve a contagem da linha especificada pelo argumento 'row'."""
       return int(self.board[row][10])
   
   def fill_row_water(self, row: int):
      """Preenche a linha especificada pelo argumento 'row' com àgua nos lugares livres."""
      for i in range(10):
         if self.board[row][i] == '':
            self.board[row][i] = WATER

   def fill_col_water(self, col: int):
      """Preenche a coluna especificada pelo argumento 'col' com àgua nos lugares livres."""
      for i in range(10):
         if self.board[i][col] == '':
            self.board[i][col] = WATER

   def is_full_col(self, col: int) -> bool:
      """Verifica se a coluna especificada pelo argumento 'col' está cheia."""
      boats = 0
      for i in range(10):
         if (self.board[i][col] != '') and (self.board[i][col] != WATER):
            boats += 1
      
      if boats == self.get_col_count(col):
         return True
      else:
         return False
   
   def is_full_row(self, row: int) -> bool:
      """Verifica se a linha especificada pelo argumento 'row' está cheia."""
      boats = 0
      for i in range(10):
         if (self.board[row][i] != '') and (self.board[row][i] != WATER):
            boats += 1
      
      if boats == self.get_row_count(row):
         return True
      else:
         return False

   def fill_water(self):
      """ Preenche as posições que só podem ter àgua"""
      #TODO: verificar posição antes de meter agua  
      same = True
      while same:
         for i in range(10):
            row_sum = np.sum(self.board[0][i])
            col_sum = np.sum(self.board[0][:][i])
            if (row_sum == self.board[2][i][9]):
               self.board[1][i] = np.zeros(10, dtype=int)
               same = False
            if (col_sum == self.board[2][i][0]):
               self.board[1][:][i] = np.zeros(10, dtype=int)
               same = False
      
      # Mete agua a toda a volta dos circulos

   @staticmethod
   def parse_instance():
      """Lê o test do standard input (stdin) que é passado como argumento
      e retorna uma instância da classe Board.
      formato do input:
      1. ROW <count-0> ... <count-9>
      2. COLUMN <count-0> ... <count-9>
      3. <hint total>
      4. HINT <row> <column> <hint value>
      """

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
      
      #adding boats to board
      for row in range (10):
         for col in range (10):
            if original_board[row][col] == 'W':
               board[1][row][col] = WATER
            elif (original_board[row][col] != ''):
               board[0][row][col] = BOAT
               if original_board[i][col] == 'T':
                  board[0][row + 1][col] = BOAT
                  if (row == 9):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                  elif (col == 0):
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                  elif (col == 9):
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row][col + 1] = WATER

               elif original_board[row][col] == 'B':
                  board[0][row - 1][col] = BOAT
                  if (row == 9):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                  elif (col == 0):
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                  elif (col == 9):
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row][col + 1] = WATER
   
               elif original_board[row][col] == 'R':
                  board[0][row][col - 1] = BOAT
                  if (row == 9):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                  elif (col == 0):
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                  elif (col == 9):
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row][col + 1] = WATER

               elif original_board[row][col] == 'L':
                  board[0][row][col + 1] = BOAT
                  if (row == 9):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                  elif (col == 0):
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                  elif (col == 9):
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                  else:
                     board[1][row][col - 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row][col + 1] = WATER
                  
               elif original_board[row][col] == 'C':
                  if (row == 0):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                  elif (row == 9):
                     if (col == 0):
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row + 1][col + 1] = WATER
                     elif (col == 9):
                        board[1][row][col - 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                     else:
                        board[1][row][col - 1] = WATER
                        board[1][row][col + 1] = WATER
                        board[1][row - 1][col] = WATER
                        board[1][row - 1][col - 1] = WATER
                        board[1][row - 1][col + 1] = WATER
                  elif (col == 0):
                     board[1][row][col + 1] = WATER
                     board[1][row + 1][col] = WATER
                     board[1][row - 1][col] = WATER
                     board[1][row - 1][col + 1] = WATER
                     board[1][row + 1][col + 1] = WATER
                  elif (col == 9):
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
                     board[1][row - 1][col + 1]
                     board[1][row - 1][col - 1]
                     board[1][row + 1][col - 1]
                     board[1][row - 1][col + 1]

      return Board(board)

      # TODO: outros metodos da classe


class Bimaru(Problem):
   def __init__(self, board: Board):
      """O construtor especifica o estado inicial."""
      self.initial = BimaruState(board)
      # TODO
      pass

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
      # TODO
      pass

   def h(self, node: Node):
      """Função heuristica utilizada para a procura A*."""
      # TODO
      pass

   # TODO: outros metodos da classe


if __name__ == "__main__":

   # TODO:
   # Ler o ficheiro do standard input,
   # Usar uma técnica de procura para resolver a instância,
   # Retirar a solução a partir do nó resultante,
   # Imprimir para o standard output no formato indicado.
   board = Board.parse_instance()
   board.fill_water()
   print(board.board)
   