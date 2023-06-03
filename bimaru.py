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
TEMP = 8
EMPTY = 0


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

   def empty_cells(self) -> int:
      """Verifica se o tabuleiro não tem pos vazias"""
      board = self.board.board[:-1, :-1]
      empty = np.count_nonzero(board == 0)
      return empty
   
   def counts_zero(self) -> bool:
      """Verifica se todas as contagens são zero"""
      for i in range(10):
         if self.board.get_row_count(i) != 0 or self.board.get_col_count(i) != 0:
            return False
      return True
   
   def has_all_boats(self) -> bool:
      """Verifica se o tabuleiro tem todos os barcos colocados"""
      return self.board.all_boats()
   
   def try_couracado(self, row: int, col: int):
      """Verifica se é possível colocar um couraçado na posição especificada pelos argumentos 'row' e 'col'."""
      pos = self.board.get_value(row, col)
      valid_positions = []

      if self.empty_cells() == 0:
         return False
      if self.board.get_row_count(row) < 4 and self.board.get_col_count(col) < 4:
         return False
      
      if pos == TOP and row + 3 < 10 and self.board.get_value(row + 1, col) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(row + 2, col) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(row + 3, col) in {BOTTOM, TEMP, EMPTY}:
         valid_positions.append((row, col, TOP))
         valid_positions.append((row + 1, col, MIDDLE))
         valid_positions.append((row + 2, col, MIDDLE))
         valid_positions.append((row + 3, col, BOTTOM))

      if pos == BOTTOM and row - 3 >= 0 and self.board.get_value(row - 1, col) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(row - 2, col) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(row - 3, col) in {TOP, TEMP, EMPTY}:
         valid_positions.append((row, col, BOTTOM))
         valid_positions.append((row - 1, col, MIDDLE))
         valid_positions.append((row - 2, col, MIDDLE))
         valid_positions.append((row - 3, col, TOP))

      if pos == LEFT and col + 3 < 10 and self.board.get_value(row, col + 1) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(row, col + 2) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(row, col + 3) in {RIGHT, TEMP, EMPTY}:
         valid_positions.append((row, col, LEFT))
         valid_positions.append((row, col + 1, MIDDLE))
         valid_positions.append((row, col + 2, MIDDLE))
         valid_positions.append((row, col + 3, RIGHT))

      if pos == RIGHT and col - 3 <= 0 and self.board.get_value(row, col - 1) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(row, col - 2) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(row, col - 3) in {LEFT, TEMP, EMPTY}:
         valid_positions.append((row, col, RIGHT))
         valid_positions.append((row, col - 1, MIDDLE))
         valid_positions.append((row, col - 2, MIDDLE))
         valid_positions.append((row, col - 3, LEFT))

      if pos == EMPTY and row + 3 < 10 and self.board.get_value(row + 1, col) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(row + 2, col) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(row + 3, col) in {BOTTOM, TEMP, EMPTY}:
         valid_positions.append((row, col, TOP))
         valid_positions.append((row + 1, col, MIDDLE))
         valid_positions.append((row + 2, col, MIDDLE))
         valid_positions.append((row + 3, col, BOTTOM))
      
      if pos == EMPTY and col + 3 < 10 and self.board.get_value(col + 1, row) in {MIDDLE, TEMP, EMPTY} and \
            self.board.get_value(col + 2, row) in {EMPTY, TEMP, MIDDLE} and \
            self.board.get_value(col + 3, row) in {RIGHT, TEMP, EMPTY}:
         valid_positions.append((row, col, LEFT))
         valid_positions.append((row, col + 1, MIDDLE))
         valid_positions.append((row, col + 2, MIDDLE))
         valid_positions.append((row, col + 3, RIGHT))

      return valid_positions
      
   def do_action(self, action: tuple):
      new_board = self.board.copy()
      new_board.set_value(action[0], action[1], action[2])
      return BimaruState(new_board)

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

   def copy(self):
      return Board(self.board.copy())

   def update_pos(self):
      row_idx, col_idx = np.where(self.board == TEMP)

      for row, col in zip(row_idx, col_idx):
         if self.adjacent_vertical_values(row, col) == (TOP, WATER):
            self.set_value(row, col, BOTTOM)
         elif self.adjacent_vertical_values(row, col) == (WATER, BOTTOM):
            self.set_value(row, col, TOP)
         elif self.adjacent_horizontal_values(row, col) == (LEFT, WATER):
            self.set_value(row, col, RIGHT)
         elif self.adjacent_horizontal_values(row, col) == (WATER, RIGHT):
            self.set_value(row, col, LEFT)
         elif self.adjacent_vertical_values(row, col) == (TOP, BOTTOM):
            self.set_value(row, col, MIDDLE)
         elif self.adjacent_horizontal_values(row, col) == (LEFT, RIGHT):
            self.set_value(row, col, MIDDLE)
         elif self.adjacent_vertical_values(row, col) == (WATER, WATER) and self.adjacent_horizontal_values(row, col) == (WATER, WATER):
            self.set_value(row, col, CIRCLE)


   def calculate_state(self):
      """Ve quantas posicoes estão preenchidas em cada linha e cada coluna e atualiza a contagem"""

      self.complete_pos()

      print('completa pos: ','\n' , self.board, '\n')

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

      # print('atualiza contagem: ','\n' , self.board, '\n')
      self.fill_water()
      # print('mete agua: ','\n' , self.board, '\n')

      self.update_pos()
      # print('atualiza pos: ','\n' , self.board, '\n')


      # calcula o numero de submarinos
      self.num_boats['submarino'] = np.count_nonzero(self.board == 7)

      # calcula o numero de cruzadores
      for row in range(10):
         for col in range(10):
            if self.board[row, col] == LEFT and self.board[row, col + 1] == RIGHT:
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


      return self

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
   
   # def get_adj_couracado(self, row: int, col: int, val: int) -> tuple:
   #    if val == TOP:
   #       return self.board[row + 1][col], self.board[row + 2][col], self.board[row + 3][col]
   #    elif val == BOTTOM:
   #       return self.board[row - 1][col], self.board[row - 2][col], self.board[row - 3][col]
   #    elif val == LEFT:
   #       return self.board[row][col + 1], self.board[row][col + 2], self.board[row][col + 3]
   #    elif val == RIGHT:
   #       return self.board[row][col - 1], self.board[row][col - 2], self.board[row][col - 3]
   #    elif val == MIDDLE:
   #       return self.board[row - 1][col], self.board[row + 1][col], 
   
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

   

   def complete_pos(self):
      """Adiciona uma posição às pontas dos barcos e atualiza as contagens"""

      for i in range(10):
         for j in range(10):
            if self.get_value(i, j) == TOP:
               self.set_value(i + 1, j, TEMP)
               # self.set_value(10, j, self.get_col_count(j) - 1)
               # self.set_value(i + 1, 10, self.get_row_count(i + 1) - 1)
            elif self.get_value(i, j) == BOTTOM:
               self.set_value(i - 1, j, TEMP)
               # self.set_value(10, j, self.get_col_count(j) - 1)
               # self.set_value(i - 1, 10, self.get_row_count(i - 1) - 1)
            elif self.get_value(i, j) == LEFT:
               self.set_value(i, j + 1, TEMP)
               # self.set_value(i, 10, self.get_row_count(i) - 1)
               # self.set_value(10, j + 1, self.get_col_count(j + 1) - 1)
            elif self.get_value(i, j) == RIGHT:
               self.set_value(i, j - 1, TEMP)
               # self.set_value(i, 10, self.get_row_count(i) - 1)
               # self.set_value(10, j - 1, self.get_col_count(j - 1) - 1)

   @staticmethod
   def parse_instance():
      """Lê o test do standard input (stdin) que é passado como argumento
      e retorna uma instância da classe Board.
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

      return Board(board).calculate_state()

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
      
      actions = []

      if state.empty_cells() == 0:
         return []
      if state.board.num_boats['couracado'] == 0:
         # tenta colocar um couracado
         for row in range(10):
            for col in range(10):
               valid_positions = state.try_couracado(row, col)
               if valid_positions:
                  actions.extend(valid_positions)
      elif state.board.num_boats['cruzador'] < 2:
         pass
                  
      return actions


   def result(self, state: BimaruState, action):
      """Retorna o estado resultante de executar a 'action' sobre
      'state' passado como argumento. A ação a executar deve ser uma
      das presentes na lista obtida pela execução de
      self.actions(state)."""
      
      return state.do_action(action)
      

   def goal_test(self, state: BimaruState):
      """Retorna True se e só se o estado passado como argumento é
      um estado objetivo. Deve verificar se todas as posições do tabuleiro
      estão preenchidas de acordo com as regras do problema."""
      return state.empty_cells() == 0 and state.counts_zero() and state.has_all_boats()

   def h(self, node: Node):
      """Função heuristica utilizada para a procura A*."""
      state = node.state
      res = state.empty_cells()

      return res


   # TODO: outros metodos da classe


if __name__ == "__main__":

   # TODO:
   # Ler o ficheiro do standard input,
   # Usar uma técnica de procura para resolver a instância,
   # Retirar a solução a partir do nó resultante,
   # Imprimir para o standard output no formato indicado.
   
   board = Board.parse_instance()
   problem = Bimaru(board)
   goal_node = depth_first_tree_search(problem)
   print(goal_node.state.board.print(), sep='')
   # test try_couracado function
   # problem.initial.try_couracado(0, 0)
   # print(board.board)
   # print('\n')
   #print the board
   # for row in board.board:
   #    print(' '.join(format(cell, '<1') for cell in row))
   
   # print(board.num_boats)
