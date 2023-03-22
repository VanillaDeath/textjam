from __future__ import annotations
import sys
import re
import operator
import typing
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit import PromptSession
from prompt_toolkit import HTML

class InvalidInput(Exception):
    def __init__(self, message: str = "Input is invalid") -> None:
        self.message = message
        super().__init__(self.message)

class InvalidOperator(Exception):
    def __init__(self, operator: str, message: str = "Invalid operator use") -> None:
        self.operator: str = operator
        self.message: str = message
        super().__init__(f"{self.message}: {self.operator}")

class SimplificationError(Exception):
    def __init__(self, tokens: list, message: str = "Error occurred while simplifying expression") -> None:
        self.tokens: list = tokens
        self.token_string: str = '[%s]' % ', '.join(map(str, self.tokens))
        self.message: str = message
        super().__init__(f"{self.message}: {self.token_string}")

class ParenError(Exception):
    def __init__(self, paren: str, message: str = "Unmatched parenthesis in expression") -> None:
        self.paren: str = paren
        self.message: str = message
        super().__init__(f"{self.message}: {self.paren}")

class TooManyDecimals(Exception):
    def __init__(self, value: str, message: str = "Too many decimal points") -> None:
        self.value: str = value
        self.message: str = message
        super().__init__(f"{self.message}: {self.value}")

class Calc:
    valid: re.Pattern = re.compile(r'^[0-9-.()+*/\\%\s^]+$')
    paren: re.Pattern = re.compile(r'\([^\(\)]+\)')
    paren_mult: tuple[re.Pattern, ...] = (
        re.compile(r'([\d.\)]+)(\()'),
        re.compile(r'(\))([\d.]+)')
        )
    operations: dict[str, typing.Callable] = {
        '^': operator.pow, '/': operator.truediv, '\\': operator.floordiv,
        '%': operator.mod, '*': operator.mul, '+': operator.add, '-': operator.sub
        }

    def __init__(self) -> None:
        """Initizalize object"""
        self.last_result: float = 0
        self.error: bool = False
        self.stale: bool = True

    def __enter__(self) -> Calc:
        self.run: bool = True
        return self

    def __exit__(self, *a) -> None:
        pass

    def calc(self, expr: str, is_sub_expr: bool = False) -> float:
        """Perform calculation on an expression"""
        try:
            if not is_sub_expr:
                if not Calc.valid.match(expr):
                    raise InvalidInput()
                p: re.Pattern
                # Apply implicit multiply for parens
                # (X)(Y) => (X)*(Y)   X(Y) => X*(Y)   (X)Y => (X)*Y
                for p in Calc.paren_mult:
                    expr = p.sub(r'\1*\2', expr)

            paren_match: re.Match | None
            # Recursively handle paren grouping
            if paren_match := Calc.paren.search(expr):
                sub_expr: str = paren_match.group()
                # (9+5)*3 => 14*3   ( replace  (9+5)          with result of 9+5 )
                return self.calc(expr.replace(sub_expr, str(self.calc(sub_expr[1:-1], is_sub_expr=True))))
            #

            # Remove whitespace characters
            # '-12+50 - 3 * 6' => '-12+50-3*6'
            # Tokenize
            # ['', '-', '12', '+', '50', '-', '3', '*', '6']
            # Remove empty tokens
            # ['-', '12', '+', '50', '-', '3', '*', '6']
            tokens: list[str] = list(filter(lambda x: x != '', re.split(
                r'([^\d.]{1})', re.sub(r'\s', r'', expr))))

            # Prohibit ends with operator
            if tokens[-1] in Calc.operations:
                raise InvalidOperator(tokens[-1], "Must not end with operator")

            k: int
            token: str
            tokens2: list[str] = []
            # For each token
            for k, token in enumerate(tokens):
                # If not an operator
                if token not in Calc.operations:
                    # Prohibit multiple decimal points
                    if token.count('.') > 1:
                        raise TooManyDecimals(token)
                    # Prohibit unmatched parens
                    if token in ('(', ')'):
                        raise ParenError(token)
                    # Convert negative numbers to negative instead of treating as subtraction
                    # ['-12', '+', '50', '-', '3', '*', '6']
                    # Not first token, and follows a '-', at start of expression or after an operator
                    if k > 0 and tokens[k-1] == '-' and (k == 1 or tokens[k-2] in Calc.operations):
                        # Change subtraction to negative number
                        tokens2[-1] += token
                        continue
                tokens2.append(token)
                # Prohibit two consecutive operators
                # if k > 1 and all(t in Calc.operations for t in tokens2[-3:-1]):
                if k > 1 and tokens2[-2] in Calc.operations and tokens2[-3] in Calc.operations:
                    raise InvalidOperator(tokens2[-3] + tokens2[-2], "Too many operators")

            # Prohibit starts with operator
            if tokens2[0] in Calc.operations:
                raise InvalidOperator(tokens2[0], "Must not begin with operator")

            # Start simplifying expression
            op: str
            operation: typing.Callable
            # Go through operators by proper order of operations
            for op, operation in Calc.operations.items():
                # While we have a match on this operator
                while op in tokens2:
                    # Operator index/position
                    # '3', '*', '6'
                    pos: int = tokens2.index(op)
                    # Replace operator with result from operation performed on values before and after it
                    # '3', '18', '6'
                    tokens2[pos] = operation(float(tokens2[pos-1]), float(tokens2[pos+1]))
                    # Remove value after
                    # '3', '18'
                    tokens2.pop(pos+1)
                    # Remove value before
                    # '18'
                    tokens2.pop(pos-1)

            # If we're not left with a single value after the process, something went wrong
            if len(tokens2) != 1:
                raise SimplificationError(tokens2)

            if not is_sub_expr:
                self.stale = False
                self.error = False

            return float(tokens2[0])
        
        except Exception as err:
            self.stale = True
            self.error = True
            print(HTML(f"<ansired>{err}</ansired>"))
            return self.last_result

    def do(self, inp: str) -> None:
        """Parse input"""
        inp = inp.strip().lower()
        match inp:
            case 'exit' | 'quit':
                self.run = False
            case '':
                print(HTML(f'<ansibrightblack>{self.last_result:g}</ansibrightblack>'))
            case _:
                try:
                    self.last_result = self.calc(inp)
                except Exception as err:
                    self.error = True
                    print(HTML(f"<ansired>{err}</ansired>"))
                deco = 'ansibrightblack' if self.stale else 'bold'
                print(HTML(f'<{deco}>{self.last_result:g}</{deco}>'))
                self.stale = True
                

def main(args: list) -> int:
    """Main routine"""

    with Calc() as calculator:

        if not calculator.run:
            return 1

        if len(args) > 1:
            calculator.do(' '.join(args[1:]))
            return 1 if calculator.error else 0

        print("+ add  - subtract  * multiply  / divide  \ foor-divide  % modulus  ^ power  () group")

        session: PromptSession = PromptSession()

        while calculator.run:
            calculator.do(session.prompt("> "))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))