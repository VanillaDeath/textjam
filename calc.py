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
    paren_cozy: tuple[re.Pattern, ...] = (
        re.compile(r'([\d.\)]+)(\()'),
        re.compile(r'(\))([\d.]+)')
        )
    operations: dict[str, typing.Callable] = {
        '^': operator.pow, '/': operator.truediv, '\\': operator.floordiv,
        '%': operator.mod, '*': operator.mul, '+': operator.add, '-': operator.sub
        }

    def __init__(self) -> None:
        """Initizalize object"""
        pass

    def __enter__(self) -> Calc:
        return self

    def __exit__(self, *a) -> None:
        pass

    def calculate(self, expr: str, is_sub_expr: bool = False) -> float | None:
        """Perform calculation on an expression"""
        try:
            # Only run once
            if not is_sub_expr:
                # Contains invalid characters
                if not Calc.valid.match(expr):
                    raise InvalidInput()
                p: re.Pattern
                # Apply implicit multiply for cozy parens
                # (X)(Y) => (X)*(Y)   X(Y) => X*(Y)   (X)Y => (X)*Y
                for p in Calc.paren_cozy:
                    expr = p.sub(r'\1*\2', expr)

            paren_match: re.Match | None
            # Recursively handle paren grouping
            if paren_match := Calc.paren.search(expr):
                sub_expr: str = paren_match.group()
                # (9+5)*3 => 14*3        ( replace  (9+5)                 with result of 9+5 )
                return self.calculate(expr.replace(sub_expr, str(self.calculate(sub_expr[1:-1], is_sub_expr=True))), is_sub_expr=True)
            #

            # Remove whitespace characters
            # '-12+50 - 3 * 6' => '-12+50-3*6'
            # Tokenize
            # ['', '-', '12', '+', '50', '-', '3', '*', '6']
            # Remove empty tokens
            # ['-', '12', '+', '50', '-', '3', '*', '6']
            tokens: list[str] = list(filter(lambda x: x != '', re.split(
                r'([^\d.]{1})', re.sub(r'\s', r'', expr))))
            tokens2: list[str] = []

            # Prohibit ends with operator
            if tokens[-1] in Calc.operations:
                raise InvalidOperator(tokens[-1], "Must not end with operator")

            k: int
            token: str
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
                # Check for two consecutive operators
                if k > 1 and tokens2[-3] in Calc.operations and tokens2[-2] in Calc.operations:
                    match tokens2[-3] + tokens2[-2]:
                        # Support python-style power operator
                        case '**':
                            tokens2[-3] = '^'
                            del tokens2[-2]
                        # Support python-style floor-divide operator
                        case '//':
                            tokens2[-3] = '\\'
                            del tokens2[-2]
                        # Otherwise prohibit two consecutive operators
                        case _:
                            raise InvalidOperator(
                                tokens2[-3] + tokens2[-2], "Multiple consecutive operators not allowed")

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
                    # Replace operator with result from operation performed on operands before and after it
                    # '3', '18', '6'
                    tokens2[pos] = operation(float(tokens2[pos-1]), float(tokens2[pos+1]))
                    # Remove operand after
                    # '3', '18'
                    del tokens2[pos+1]
                    # Remove operand before
                    # '18'
                    del tokens2[pos-1]

            # If we're not left with a single value after the process, something went wrong
            if len(tokens2) != 1:
                raise SimplificationError(tokens2)

            return float(tokens2[0])

        except Exception as err:
            raise err

    def do(self, inp: str) -> dict:
        """Parse input"""
        result: float | None = None
        error: str | None = None
        do_exit: bool = False
        match inp.strip().lower():
            case 'exit' | 'quit':
                do_exit = True
            case '':
                pass
            case _ as expr:
                try:
                    result = self.calculate(expr)
                except Exception as err:
                    error = str(err)
        return {'result': result, 'error': error, 'exit': do_exit}

def main(args: list) -> int:
    """Main routine"""

    calculator: Calc
    with Calc() as calculator:

        result: dict

        if len(args) > 1:
            result = calculator.do(' '.join(args[1:]))
            if result['error'] is not None:
                print(HTML(f"<ansired>{result['error']}</ansired>"))
                return 1
            print(HTML(f"<bold>{result['result']:g}</bold>"))
            return 0

        print("+ add  - subtract  * multiply  / divide  \ foor-divide  % modulus  ^ power  () group")

        session: PromptSession = PromptSession()

        while result := calculator.do(session.prompt("> ")):
                if result['exit']:
                    break
                if result['error'] is not None:
                    print(HTML(f"<ansired>{result['error']}</ansired>"))
                    continue
                if result['result'] is not None:
                    print(HTML(f"<bold>{result['result']:g}</bold>"))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))