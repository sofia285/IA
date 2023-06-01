# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 72:
# 102835 Sofia Paiva
# 102904 Mariana Miranda

WATER = 1
CIRCLE = 7
TOP = 2
MIDDLE = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6


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

   def is_complete(self) -> bool:
      return self.board.all_boats()   
   
   def counts_zero(self) -> bool:
      """Verifica se todas as contagens são zero"""
      for i in range(10):
         if self.board.get_row_count(i) != 0 or self.board.get_col_count(i) != 0:
            return False
      return True
   
   # TODO: outros metodos da classe


class Board:
   """Representação interna de um tabuleiro de Bimaru."""

   num_boats = {
      'couracado': 0, # 1
      'cruzador': 0, # 2
      'contratorpedeiro': 0, # 3
      'submarino': 0, # 4
   }

   def __init__(self, board: np.ndarray):
      """Construtor da classe"""
      self.board = board

   def calculate_state(self):
      """Ve quantas posicoes estão preenchidas em cada linha e cada coluna e atualiza a contagem"""

      # calcular o numero de cada barco
      # calcula o numero de submarinos
      self.num_boats['submarino'] = np.count_nonzero(self.board == 7)

      # calcula o numero de cruzadores
      for row in range(10):
         for col in range(10):
            if self.board[row, col] == 5 and self.board[row, col + 1] == 6:
               self.num_boats['cruzador'] += 1
            elif self.board[row, col] == 2 and self.board[row + 1, col] == 4:
               self.num_boats['cruzador'] += 1
      
      # calcula o numero de contratorpedeiros
      for row in range(10):
         for col in range(10):
            if self.board[row, col] == 5 and self.board[row, col + 1] == 3 and self.board[row, col + 2] == 6:
               self.num_boats['contratorpedeiro'] += 1
            elif self.board[row, col] == 2 and self.board[row + 1, col] == 3 and self.board[row + 2, col] == 4:
               self.num_boats['contratorpedeiro'] += 1
      
      # calcula o numero de couracados
      for row in range(10):
         for col in range(10):
            if self.board[row, col] == 5 and self.board[row, col + 1] == 3 and self.board[row, col + 2] == 3 and self.board[row, col + 3] == 6:
               self.num_boats['couracado'] += 1
            elif self.board[row, col] == 2 and self.board[row + 1, col] == 3 and self.board[row + 2, col] == 3 and self.board[row + 3, col] == 4:
               self.num_boats['couracado'] += 1

      # Atualiza as contagens   	
      for i in range(10):
         row_count = 0
         for j in range(10):
            if self.get_value(i, j) != 0 and self.get_value(i, j) != 1:
               row_count += 1
         self.set_value(i, 10, self.get_row_count(i) - row_count)

      for i in range(10):
         col_count = 0
         for j in range(10):
            if self.get_value(j, i) != 0 and self.get_value(j, i) != 1:
               col_count += 1
         self.set_value(10, i, self.get_col_count(i) - col_count)

   def all_boats(self):
      return self.num_boats['couracado'] == 1 and self.num_boats['cruzador'] == 2 and self.num_boats['contratorpedeiro'] == 3 and self.num_boats['submarino'] == 4

   def get_value(self, row: int, col: int) -> str:
      """Devolve o valor na respetiva posição do tabuleiro."""
      return self.board[row][col]

   def adjacent_vertical_values(self, row: int, col: int) -> tuple:#(str, str) estava a dar erro
      """Devolve os valores imediatamente acima e abaixo,
      respectivamente."""
      return self.board[row - 1][col], self.board[row + 1][col]

   def adjacent_horizontal_values(self, row: int, col: int) -> tuple:
      """Devolve os valores imediatamente à esquerda e à direita,
      respectivamente."""
      return self.board[row][col - 1], self.board[row][col + 1]

   def get_row(self, row: int) -> np.ndarray:
      """Devolve a linha especificada pelo argumento 'row'."""
      return self.board[row]
   
   def get_column(self, col: int) -> np.ndarray:
      """Devolve a coluna especificada pelo argumento 'col'."""
      return self.board[:, col]
   
   def set_value(self, row: int, col: int, value: int):
      """Atribui o valor 'value' à posição especificada pelos
      argumentos 'row' e 'col'."""
      self.board[row][col] = value

   def get_col_count(self, col: int) -> int:
     """Deveolve a contagem da coluna especificada pelo argumento 'col'."""
     return self.board[10][col]
   
   def get_row_count(self, row: int) -> int:
       """Deveolve a contagem da linha especificada pelo argumento 'row'."""
       return self.board[row][10]
   
   def fill_row_water(self, row: int):
      """Preenche a linha especificada pelo argumento 'row' com àgua nos lugares livres."""
      for i in range(10):
         if self.board[row][i] == 0:
            self.board[row][i] = WATER

   def fill_col_water(self, col: int):
      """Preenche a coluna especificada pelo argumento 'col' com àgua nos lugares livres."""
      for i in range(10):
         if self.board[i][col] == 0:
            self.board[i][col] = WATER

   def fill_water(self):
      """ Preenche as posições que só podem ter àgua"""
      #TODO: verificar posição antes de meter agua

      for i in range(10):
         if self.get_row_count(i) == 0:
            self.fill_row_water(i)
      
         elif self.get_col_count(i) == 0:
            self.fill_col_water(i)   
      
      # Mete agua a toda a volta dos circulos
      for i in range(10):
        for j in range(10):
            if self.get_value(i, j) == CIRCLE:
               for di in range(-1, 2):
                  for dj in range(-1, 2):
                     if di == 0 and dj == 0:
                        continue
                     if 0 <= i + di < 10 and 0 <= j + dj < 10:
                        self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == TOP:
               for di in range(-1, 2):
                  for dj in range(-1, 2):
                        if (di == 0 and dj == 0) or (di == 1 and dj == 0):
                           continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 10:
                           self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == BOTTOM:
               for di in range(-1, 2):
                  for dj in range(-1, 2):
                     if (di == 0 and dj == 0) or (di == -1 and dj == 0):
                        continue
                     if 0 <= i + di < 10 and 0 <= j + dj < 10:
                        self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == LEFT:
               for di in range(-1, 2):
                  for dj in range(-1, 2):
                        if (di == 0 and dj == 0) or (di == 0 and dj == 1):
                           continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 10:
                           self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == RIGHT:
               for di in range(-1, 2):
                  for dj in range(-1, 2):
                        if (di == 0 and dj == 0) or (di == 0 and dj == -1):
                           continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 10:
                           self.set_value(i + di, j + dj, WATER)

   def try_couracado(self, row: int, col: int) -> bool:
      """Verifica se é possível colocar um couraçado na posição especificada pelos argumentos 'row' e 'col'."""
      if self.get_value(row, col) != '':
         return False
      if self.get_row_count(row) < 4 or self.get_col_count(col) < 4:
         return False
      return True
      #falta mt coisa

   def complete_pos(self):
      """Adiciona uma posição às pontas dos barcos e atualiza as contagens"""

      for i in range(10):
         for j in range(10):
            if self.get_value(i, j) == TOP:
               self.set_value(i + 1, j, MIDDLE)
               self.set_value(10, j, self.get_col_count(j) - 1)
               self.set_value(i + 1, 10, self.get_row_count(i + 1) - 1)
            elif self.get_value(i, j) == BOTTOM:
               self.set_value(i - 1, j, MIDDLE)
               self.set_value(10, j, self.get_col_count(j) - 1)
               self.set_value(i - 1, 10, self.get_row_count(i - 1) - 1)
            elif self.get_value(i, j) == LEFT:
               self.set_value(i, j + 1, MIDDLE)
               self.set_value(i, 10, self.get_row_count(i) - 1)
               self.set_value(10, j + 1, self.get_col_count(j + 1) - 1)
            elif self.get_value(i, j) == RIGHT:
               self.set_value(i, j - 1, MIDDLE)
               self.set_value(i, 10, self.get_row_count(i) - 1)
               self.set_value(10, j - 1, self.get_col_count(j - 1) - 1)

   @staticmethod
   def parse_instance():
      """Lê o test do standard input (stdin) que é passado como argumento
      e retorna uma instância da classe Board.
      input:
      ROW 2 3 2 2 3 0 1 3 2 2
      COLUMN 6 0 1 0 2 1 3 1 2 4
      6
      HINT 0 0 T
      HINT 1 6 M
      HINT 3 2 C
      HINT 6 0 W
      HINT 8 8 B
      HINT 9 5 C
      """

      row_line = stdin.readline().rstrip('\n').split('\t')
      row = [int(x) for x in row_line[1:]]

      column_line = stdin.readline().rstrip('\n').split('\t')
      column = [int(x) for x in column_line[1:]]
      num_hints = int(input())

      board = np.zeros((11, 11), dtype=int)
      board[:-1, -1] = row
      board[-1, :-1] = column

      hint = None
      for i in range(num_hints):
         hint_line = stdin.readline().rstrip('\n').split('\t')
         if hint_line[3] == 'W':
            hint = 1
         elif hint_line[3] == 'C':
            hint = 7
         elif hint_line[3] == 'T':
            hint = 2
         elif hint_line[3] == 'M':
            hint = 3
         elif hint_line[3] == 'B':
            hint = 4
         elif hint_line[3] == 'L':
            hint = 5
         elif hint_line[3] == 'R':
            hint = 6
         hint_row = int(hint_line[1])
         hint_column = int(hint_line[2])
         board[hint_row][hint_column] = hint

         
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
      return state.is_complete() and state.counts_zero()

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
   for i in range(10):
      board.calculate_state()
      print(board.board)
      print('\n')
      board.fill_water()
      print(board.board)
      print('\n')
      board.complete_pos()
      #print the board
      # for row in board.board:
      #    print(' '.join(format(cell, '<1') for cell in row))
      print(board.board)
      print('\n')
      print(board.num_boats)
