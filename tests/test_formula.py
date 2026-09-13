import importlib.util
import math
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('formula', Path(__file__).parents[1]/'tools/formula.py')
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

class FormulaTests(unittest.TestCase):
    def test_values(self):
        f=formula.expression('log(cos(y)/cos(x))+sqrt(abs(x))', {'x','y'})
        self.assertAlmostEqual(f(x=.2,y=.4), math.log(math.cos(.4)/math.cos(.2))+math.sqrt(.2))
    def test_unsafe_syntax(self):
        for text in ['__import__("os")', 'x.__class__', '[x for x in [1]]', 'open("file")', 'sin(x, y)', 'unknown+1']:
            with self.subTest(text=text), self.assertRaises(ValueError): formula.expression(text, {'x','y'})
    def test_invalid_domain(self):
        for text in ['1/0','sqrt(-1)','exp(1000)','2**100000']:
            with self.subTest(text=text), self.assertRaises((ValueError,ZeroDivisionError,OverflowError)):formula.expression(text, set())()

if __name__=='__main__': unittest.main()
