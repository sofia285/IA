# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 72:
# 102835 Sofia Paiva
# 102904 Mariana Miranda

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

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        formato do input:
        1. ROW <count-0> ... <count-9>
        2. COLUMN <count-0> ... <count-9>
        3. <hint total>
        4. HINT <row> <column> <hint value>

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        # row = stdin.readline().rstrip('\n').split('\t')
        # columns = stdin.readline().rstrip('\n').split('\t')
        # hints = stdin.readline().rstrip('\n').split('\t')
        # print(row)
        # print(columns)
        # print(hints)
        # i = int(hints[0])
        # while i > 0:
        #     hint = stdin.readline().rstrip('\n').split('\t')
        #     i = i - 1
        #     print(hint)


        row_line = stdin.readline().rstrip('\n').split('\t')
        row = [int(x) for x in row_line[1:]]

        column_line = stdin.readline().rstrip('\n').split('\t')
        column = [int(x) for x in column_line[1:]]
        num_hints = int(input())

        board = np.zeros((11, 11), dtype=str)
        board[0] = row
        board[:, 0] = column

        for i in range(num_hints):
            hint_line = stdin.readline().rstrip('\n').split('\t')
            hint = hint_line[3]
            hint_row = int(hint_line[1]) - 1
            hint_column = int(hint_line[2]) - 1
            board[hint_row][hint_column] = hint
        
        return Board(board)
        # TODO: Ainda não testei!!!

    # TODO: outros metodos da classe


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
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
   board = Board.parse_instance()

   # TODO:
   # Ler o ficheiro do standard input,
   # Usar uma técnica de procura para resolver a instância,
   # Retirar a solução a partir do nó resultante,
   # Imprimir para o standard output no formato indicado.
   pass