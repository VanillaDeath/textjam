from __future__ import annotations
import sys
import re
import operator
import typing
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit import PromptSession
from prompt_toolkit import HTML

class InvalidOperator(Exception):
    def __init__(self, operator: str, message: str = "Invalid operator use") -> None:
        self.operator: str = operator
        self.message: str = message
        super().__init__(f"{self.message} ({self.operator})")

class SimplificationError(Exception):
    def __init__(self, tokens: list, message: str = "Error occurred while simplifying expression") -> None:
        self.tokens: list = tokens
        self.token_string: str = '[%s]' % ', '.join(map(str, tokens))
        self.message: str = message
        super().__init__(f"{self.message}\n{self.token_string}")

class TooManyDecimals(Exception):
    def __init__(self, value: str, message: str = "Too many decimal points") -> None:
        self.value: str = value
        self.message: str = message
        super().__init__(f"{self.message} ({self.value})")

class Calc:
    valid: re.Pattern = re.compile(r'^[0-9-.()+*/\\% ^]+$')
    paren: re.Pattern = re.compile(r'\([^\(\)]+\)')
    operations: dict[str, typing.Callable] = {'^': operator.pow, '/': operator.truediv, '\\': operator.floordiv,
                                       '%': operator.mod, '*': operator.mul, '+': operator.add, '-': operator.sub}

    def __init__(self) -> None:
        """Initizalize object"""
        self.last_result: float = 0

    def __enter__(self) -> Calc:
        self.run: bool = True
        return self

    def __exit__(self, *a) -> None:
        pass

    def calc(self, expr: str) -> float:
        """Perform calculation on an expression"""
        paren_match: re.Match | None
        # Recursively handle paren grouping
        if paren_match := Calc.paren.search(expr):
            sub_expr: str = paren_match.group()
            # (9+5)*3 => 14*3   ( replace  (9+5)          with result of 9+5       )
            return self.calc(expr.replace(sub_expr, str(self.calc(sub_expr[1:-1]))))
        
        # Remove whitespace characters
        # '-12+50 - 3 * 6' => '-12+50-3*6'
        # Tokenize
        # ['', '-', '12', '+', '50', '-', '3', '*', '6']
        # Remove empty tokens
        # ['-', '12', '+', '50', '-', '3', '*', '6']
        tokens: list = list(filter(lambda x: x != '', re.split(
            r'([^\d.]{1})', expr.replace(' ', ''))))

        try:
            k: int
            token: str
            # For each token
            for k, token in enumerate(tokens):
                # Ends with operator not allowed
                if token in Calc.operations and k == len(tokens) - 1:
                    raise InvalidOperator(token)
                # Convert negative numbers to negative instead of treating as subtraction
                # Determined by '-', 'X' at start of expression or immediately following an operator
                # ['-12', '+', '50', '-', '3', '*', '6']
                if token == '-' and tokens[k+1] not in Calc.operations and (k == 0 or tokens[k-1] in Calc.operations):
                    tokens[k+1] = token + tokens[k+1]
                    tokens.pop(k)
                    continue
                # Start with operator/two consecutive operators not allowed
                if token in Calc.operations and (k == 0 or tokens[k-1] in Calc.operations):
                    raise InvalidOperator(token)
                # Prohibit multiple decimal points
                if token not in Calc.operations and token.count('.') > 1:
                    raise TooManyDecimals(token)
        except Exception as err:
            print(err)
            return self.last_result

        try:
            op: str
            operation: typing.Callable
            # Go through operators by proper order of operations
            for op, operation in Calc.operations.items():
                # While we have a match on this operator
                while op in tokens:
                    # Token index/position
                    pos: int = tokens.index(op)
                    # Replace token with result from operation performed on values before and after it
                    tokens[pos] = operation(float(tokens[pos-1]), float(tokens[pos+1]))
                    # Remove value after
                    tokens.pop(pos+1)
                    # Remove value before
                    tokens.pop(pos-1)

            # If we're not left with a single value after the process, something went wrong
            if len(tokens) != 1:
                raise SimplificationError(tokens)

            return float(tokens[0])
        except Exception as err:
            print(err)
            return self.last_result

    def do(self, inp: str) -> None:
        """Parse input"""
        inp = inp.strip().lower()
        match inp:
            case 'exit':
                self.run = False
            case '':
                print(f'{self.last_result:g}')
            case _:
                if not Calc.valid.match(inp):
                    print("Invalid input")
                else:
                    try:
                        self.last_result = self.calc(inp)
                    except ZeroDivisionError:
                        print("Can't divide by 0")
                print(f'{self.last_result:g}')
                

def main(args: list) -> int:
    """Main routine"""

    with Calc() as calculate:

        if not calculate.run:
            return 1

        print("+ add  - subtract  * multiply  / divide  \ foor-divide  % modulus  ^ power  () group")

        session: PromptSession = PromptSession()

        while calculate.run:
            inp: str = session.prompt("> ")
            calculate.do(inp)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))