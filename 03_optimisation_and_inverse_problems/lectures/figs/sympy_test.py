import sympy as sp
import graphviz

x = sp.Symbol('x')
eq = 2*x + 1

dot_str = sp.printing.dot.dotprint(eq)
dot = graphviz.Source(dot_str)
dot.render('sympy_test')

deq = eq.diff()

dot_str = sp.printing.dot.dotprint(deq)
dot = graphviz.Source(dot_str)
dot.render('sympy_test_diff')



