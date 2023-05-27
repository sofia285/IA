# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 72:
# 102835 Sofia Paiva
# 102904 Mariana Miranda

WATER = '.'
CIRCLE = 'c'
TOP = 't'
MIDDLE = 'm'
BOTTOM = 'b'
LEFT = 'l'
RIGHT = 'r'

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

   
   
   # TODO: outros metodos da classe


class Board:
   """Representação interna de um tabuleiro de Bimaru."""

   def __init__(self, board: np.ndarray):
      """Construtor da classe"""
      self.board = board


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
   
   def set_value(self, row: int, col: int, value: str):
      """Atribui o valor 'value' à posição especificada pelos
      argumentos 'row' e 'col'."""
      self.board[row][col] = value

   def get_col_count(self, col: int) -> str:
     """Deveolve a contagem da coluna especificada pelo argumento 'col'."""
     return self.board[10][col]
   
   def get_row_count(self, row: int) -> str:
       """Deveolve a contagem da linha especificada pelo argumento 'row'."""
       return self.board[row][10]
   
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

   def fill_water(self):
      """ Preenche as posições que só podem ter àgua"""
      #TODO: verificar posição antes de meter agua

      for i in range(10):
         if self.get_row_count(i) == '0':
            self.fill_row_water(i)
      
         elif self.get_col_count(i) == '0':
            self.fill_col_water(i)   
      
      # Mete agua a toda a volta dos circulos
      for i in range(10):
        for j in range(10):
            if self.get_value(i, j) == 'C':
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 9:
                            self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == 'T':
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if (di == 0 and dj == 0) or (di == 1 and dj == 0):
                            continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 9:
                            self.set_value(i + di, j + dj, WATER)
            elif self.get_value(i, j) == 'B':
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if (di == 0 and dj == 0) or (di == -1 and dj == 0):
                            continue
                        if 0 <= i + di < 10 and 0 <= j + dj < 9:
                            self.set_value(i + di, j + dj, WATER) 

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

      row_line = stdin.readline().rstrip('\n').split('\t')
      row = [int(x) for x in row_line[1:]]

      column_line = stdin.readline().rstrip('\n').split('\t')
      column = [int(x) for x in column_line[1:]]
      num_hints = int(input())

      board = np.zeros((11, 11), dtype=str)
      board[:-1, -1] = row
      board[-1, :-1] = column


      for i in range(num_hints):
         hint_line = stdin.readline().rstrip('\n').split('\t')
         hint = hint_line[3]
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
   #board.fill_water()
    #print the board
   for row in board.board:
      print(' '.join(format(cell, '<1') for cell in row))