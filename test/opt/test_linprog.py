import sys
import testoptpathmagic

from math import inf
import numpy as np
import unittest

from opt.linprog import Simplex, Tableau


class LinprogTestCase(unittest.TestCase):

    @staticmethod
    def init_infeasible_max_arrays1():
        A = np.array([[-1., 2., 3.], [0., -1., 2.], [3., 1., -1.]])
        b = np.array([5., -2., -1.])
        c = np.array([-1., 1., 0.])
        return A, b, c

    @staticmethod
    def init_problem_arrays1():
        """Example arrays and vectors from p24, Ex 2 of UCLA LP document."""
        A = np.array([[0., 1., 2.], [-1., 0., 3.], [2., 1., 1.]])
        b = np.array([3., 2., 1.])
        c = np.array([1., 1., 2.])
        return A, b, c

    @staticmethod
    def init_problem_arrays2():
        """Example arrays and vectors from p26, Ex 3 of UCLA LP document"""
        A = np.array([[0., 1., 2.], [-1., 0., -3.], [2., 1., 7.]])
        b = np.array([3., -2., 5.])
        c = np.array([1., 1., 5.])
        return A, b, c

    @staticmethod
    def init_problem_arrays3():
        """Example arrays and vectors from the dual problem of Ch 4 Exercise 1, p55 of UCLA LP document"""
        A = np.array([[1., -1., -2., -1.], [2., 0., 1., -4.], [-2., 1., 0., 1.]])
        b = np.array([4., 2., 1.])
        c = np.array([1., -2., -3., -1.])
        return A, b, c

    @staticmethod
    def init_unbounded_max_arrays1():
        """Example arrays and vectors from Exercise 3 of Ch 4 of UCLA LP document."""
        A = np.array([[-1., 3., 1.], [2., -1., -2.], [-1., 0., 1.]])
        b = np.array([3., 1., 1.])
        c = np.array([-1., -1., 2.])
        return A, b, c

    def init_tableau1(self):
        A, b, c = self.init_problem_arrays1()
        return Tableau(A, b, c)

    def test_max_infeasible(self):
        A, b, c = self.init_infeasible_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        self.assertIsNone(S.max_solution)

    def test_max_solution_when_requesting_min(self):
        A, b, c = self.init_unbounded_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='min')

        solution = inf
        self.assertAlmostEqual(solution, S.max_solution)

    def test_max_unbounded_feasible(self):
        A, b, c = self.init_unbounded_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        solution = inf
        self.assertAlmostEqual(solution, S.max_solution)

    def test_min_infeasible(self):
        A, b, c = self.init_unbounded_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='min')

        self.assertIsNone(S.min_solution)

    def test_min_solution_when_requesting_max(self):
        A, b, c = self.init_unbounded_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        self.assertIsNone(S.min_solution)

    def test_min_unbounded_feasible(self):
        A, b, c = self.init_infeasible_max_arrays1()
        S = Simplex(A, b, c, debug=False, which='min')

        solution = -inf
        self.assertAlmostEqual(solution, S.min_solution)

    def test_pivot1_A(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        Aprime = np.array([[-2., -1., 1.], [-1., 0., 3.], [2., 1., 1.]])
        self.assertTrue(np.allclose(Aprime, T.A))

    def test_pivot1_b(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        bprime = np.array([2., 2., 1.])
        self.assertTrue(np.allclose(bprime, T.b))

    def test_pivot1_c(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        cprime = np.array([-1., -1., 1.])
        self.assertTrue(np.allclose(cprime, T.c))

    def test_pivot1_r_indices(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        rprime = np.array([0, 2, 2])
        self.assertTrue(np.alltrue(np.array(T.r) == rprime))

    def test_pivot1_r_mask(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        self.assertTrue(np.alltrue(T.r.mask == np.array([1, 0, 1])))

    def test_pivot1_t_indices(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        tprime = np.array([0, 1, 1])
        self.assertTrue(np.alltrue(np.array(T.t) == tprime))

    def test_pivot1_t_mask(self):
        T = self.init_tableau1()
        T.pivot(2, 1)

        self.assertTrue(np.alltrue(T.t.mask == np.array([1, 1, 0])))

    def test_solve_max1_A(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        Asolved = np.array([[-1., -5./3., -1./3.],
                            [ 0., -1./3.,  1./3.],
                            [ 1.,  7./3., -1./3.]])
        self.assertTrue(np.allclose(Asolved, S.T.A))

    def test_solve_max1_b(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        bsolved = np.array([4./3., 2./3., 1./3.])
        self.assertTrue(np.allclose(bsolved, S.T.b))

    def test_solve_max1_c(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        csolved = np.array([-1., -2./3., -1./3.])
        self.assertTrue(np.allclose(csolved, S.T.c))

    def test_solve_max1_r_indices(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        r_indices = np.array([2, 0, 1])
        self.assertTrue(np.alltrue(np.array(S.T.r) == r_indices))

    def test_solve_max1_r_mask(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        r_mask = [0, 1, 0]
        self.assertTrue(np.alltrue(S.T.r.mask == r_mask))

    def test_solve_max1_t_indices(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        t_indices = np.array([0, 2, 1])
        self.assertTrue(np.alltrue(np.array(S.T.t) == t_indices))

    def test_solve_max1_t_mask(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        t_mask = [1, 0, 0]
        self.assertTrue(np.alltrue(S.T.t.mask == t_mask))

    def test_solve_max1_x(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        xsolved = [0., 1./3., 2./3.]
        self.assertTrue(np.allclose(xsolved, S.T.x))

    def test_solve_max1_y(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        ysolved = [0., 1./3., 1.]
        self.assertTrue(np.allclose(ysolved, S.T.y))

    def test_solve_max_solution(self):
        A, b, c = self.init_problem_arrays1()
        S = Simplex(A, b, c, debug=False, which='max')

        solution = 5./3.
        self.assertAlmostEqual(solution, S.max_solution)

    def test_solve_max2_b(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        bsolved = np.array([4./3., 2./3., 1./3.])
        self.assertTrue(np.allclose(bsolved, S.T.b))

    def test_solve_max2_t_indices(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        t_indices = np.array([0, 2, 1])
        self.assertTrue(np.alltrue(np.array(S.T.t) == t_indices))

    def test_solve_max2_t_mask(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        t_mask = [1, 0, 0]
        self.assertTrue(np.alltrue(S.T.t.mask == t_mask))

    def test_solve_max2_x(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        xsolved = [0., 1./3., 2./3.]
        self.assertTrue(np.allclose(xsolved, S.T.x))

    def test_solve_max2_y(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        ysolved = [0., 2./3., 1.]
        self.assertTrue(np.allclose(ysolved, S.T.y))

    def test_solve_max2_solution(self):
        A, b, c = self.init_problem_arrays2()
        S = Simplex(A, b, c, debug=False, which='max')

        solution = 11./3.
        self.assertAlmostEqual(solution, S.max_solution)

    def test_solve_min3_b(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        b_solved = np.array([3., 7., 12.])
        self.assertTrue(np.allclose(b_solved, S.T.b))

    def test_solve_min3_c(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        c_solved = np.array([0., -1., -1., -1.])
        self.assertTrue(np.allclose(c_solved, S.T.c))

    def test_solve_min3_r_indices(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        r_indices = [1, 1, 2, 0]
        self.assertTrue(np.alltrue(S.T.r.data == r_indices))

    def test_solve_min3_r_mask(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        r_mask = [0, 1, 1, 0]
        self.assertTrue(np.alltrue(S.T.r.mask == r_mask))

    def test_solve_min3_t_indices(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        t_indices = [3, 0, 2]
        self.assertTrue(np.alltrue(S.T.t.data == t_indices))

    def test_solve_min3_t_mask(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        t_mask = [0, 0, 1]
        self.assertTrue(np.alltrue(S.T.t.mask == t_mask))

    def test_solve_min3_x(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        x_solved = [7., 0., 0., 3.]
        self.assertTrue(np.allclose(x_solved, S.T.x))

    def test_solve_min3_y(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        y_solved = [1., 0., 0.]
        self.assertTrue(np.allclose(y_solved, S.T.y))

    def test_solve_min3_solution(self):
        A, b, c = self.init_problem_arrays3()
        S = Simplex(A, b, c, debug=False, which='min')

        solution = 4.
        self.assertAlmostEqual(solution, S.min_solution)


if __name__ == '__main__':
    unittest.main()
