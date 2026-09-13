#!/usr/bin/env python3
"""Дискретизация безопасного арифметического выражения; сторонние пакеты не нужны."""
import argparse
import ast
import math
from pathlib import Path

FUNCTIONS = {name: getattr(math, name) for name in (
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt')}
FUNCTIONS['abs'] = abs
CONSTANTS = {'pi': math.pi, 'e': math.e}


def expression(text, variables):
    tree = ast.parse(text, mode='eval')
    if len(list(ast.walk(tree))) > 200:
        raise ValueError('Слишком длинное выражение')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or len(node.args) != 1 or node.keywords:
                raise ValueError('Разрешены только перечисленные функции с одним аргументом')
        elif isinstance(node, ast.Name):
            if node.id not in set(variables) | FUNCTIONS.keys() | CONSTANTS.keys():
                raise ValueError(f'Неизвестное имя: {node.id}')
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or abs(node.value) > 1e100:
                raise ValueError('Недопустимая константа')
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                                   ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Load)):
            raise ValueError('Разрешены числа, переменные, арифметика и математические функции')
    # Собственный интерпретатор дерева: без eval, атрибутов, импортов и доступа к файлам.
    def calculate(**values):
        def visit(n):
            if isinstance(n, ast.Constant): return float(n.value)
            if isinstance(n, ast.Name): return values[n.id] if n.id in values else CONSTANTS[n.id]
            if isinstance(n, ast.Call): return FUNCTIONS[n.func.id](visit(n.args[0]))
            if isinstance(n, ast.UnaryOp): return visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div): return a/b
            if abs(b) > 1000: raise ValueError('Слишком большой показатель степени')
            return math.pow(a, b)
        result = visit(tree.body)
        if not math.isfinite(result): raise ValueError('Формула дала неконечное значение')
        return result
    return calculate


def main():
    parser = argparse.ArgumentParser(description='Формула → контур CSV или треугольная поверхность OBJ')
    sub = parser.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('contour', help='Замкнутый контур x(t), y(t), z(t)')
    for name, default in [('x', 'cos(t)'), ('y', 'sin(t)'), ('z', '0.35*cos(2*t)')]:
        c.add_argument('--'+name, default=default)
    c.add_argument('--start', type=float, default=0)
    c.add_argument('--end', type=float, default=2*math.pi)
    c.add_argument('--samples', type=int, default=64)
    c.add_argument('--output', default='contour.csv')
    s = sub.add_parser('surface', help='График z=f(x,y) на прямоугольнике')
    s.add_argument('--z', required=True)
    s.add_argument('--x-range', nargs=2, type=float, default=[-1, 1])
    s.add_argument('--y-range', nargs=2, type=float, default=[-1, 1])
    s.add_argument('--steps', type=int, default=20, help='Число ячеек по каждой оси')
    s.add_argument('--output', default='initial.obj')
    args = parser.parse_args()
    try:
        lines = []
        if args.mode == 'contour':
            if not 3 <= args.samples <= 4096 or not all(map(math.isfinite, [args.start, args.end])) or args.end <= args.start:
                raise ValueError('Нужны 3–4096 отсчётов и конечный возрастающий интервал')
            funcs = [expression(getattr(args, k), {'t'}) for k in ('x', 'y', 'z')]
            first = [f(t=args.start) for f in funcs]; last = [f(t=args.end) for f in funcs]
            points = [[f(t=args.start+(args.end-args.start)*i/args.samples) for f in funcs] for i in range(args.samples)]
            scale = max(math.dist(p, first) for p in points)
            if scale == 0 or math.dist(first, last) > 1e-8*scale:
                raise ValueError('Параметризация должна быть замкнута на концах интервала и невырождена')
            lines = ['# x,y,z; точки идут по порядку, первая не повторяется']
            lines += [','.join(format(v, '.17g') for v in p) for p in points]
        else:
            if not 2 <= args.steps <= 200: raise ValueError('steps должен быть от 2 до 200')
            for bounds in [args.x_range, args.y_range]:
                if not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]: raise ValueError('Диапазон должен быть конечным и возрастающим')
            f = expression(args.z, {'x', 'y'}); n = args.steps
            for j in range(n+1):
                y = args.y_range[0]+(args.y_range[1]-args.y_range[0])*j/n
                for i in range(n+1):
                    x = args.x_range[0]+(args.x_range[1]-args.x_range[0])*i/n
                    lines.append(f'v {x:.17g} {y:.17g} {f(x=x,y=y):.17g}')
            for j in range(n):
                for i in range(n):
                    a = j*(n+1)+i+1; b=a+1; d=a+n+1; c=d+1
                    lines += [f'f {a} {b} {c}', f'f {a} {c} {d}']
        Path(args.output).write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('Сохранён', args.output)
        if args.mode == 'surface': print('При расчёте используйте --normal 0 0 1: высоты оптимизируются над плоскостью XY.')
    except (ValueError, SyntaxError, KeyError, TypeError, ZeroDivisionError, OverflowError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
