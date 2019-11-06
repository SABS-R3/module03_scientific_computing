import sympy as sp
import graphviz

x = sp.Symbol('x')
eq = 2*x + 1

dot_str = sp.printing.dot.dotprint(eq)
dot = graphviz.Source(dot_str)
dot.render('sympy_test')

