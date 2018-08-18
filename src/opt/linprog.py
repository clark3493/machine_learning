from math import inf
import numpy as np


class Simplex(object):
    """
    An object for solving standard linear min or max problems with the Simplex method.

    A is a 2D array and b and c are interpreted as column vectors. The 'A' array
    should have the same number of columns as elements in 'c' and the same number
    of rows as elements in 'b'. The which flag can be set to 'max' or 'min'
    depending on whether the solution to the standard maximum or minimum problem
    is desired. The indicated solution type is only relevant for solution efficiency.

    Parameters
    ----------
    A: (m, n) ndarray
        Problem A array
    b: (m,) ndarray
        Problem b vector
    c: (n,) ndarray
        Problem c vector
    auto: bool, optional
        Automatically solve the problem once it is defined. Default=True.
    debug: bool, optional
        Output debug information for each solution iteration. Debug=False.
    which: {'max', 'min'}, optional
        Desired solution type. Default='max'.

    Notes
    -----
    The maximum problem is to maximize transpose(c).dot(x) subject to A.dot(x)<=b
    The minimum problem is to minimize transpose(y).dot(b) subject to
        transpose(y).dot(A) >= transpose(c)
    The standard maximum problem implies that x >= 0. Similarly for y in the standard
        minimum problem.

    There is no need to redefine the 'which' parameter if requesting the opposite
    solution value from how the Simplex object was originally defined. The 'which'
    flag is only for solution efficiency.

    See Also
    --------
    Tableau : Simplex pivot table

    References
    ----------
    [1] Ferguson, Thomas S., Linear Programming: A Concise Introduction.

    Examples
    --------
    >>> import numpy as np
    >>> A = [[0., 1., 2.], [-1., 0., -3.], [2., 1., 7.]]
    >>> b = [3., -2., 5.]
    >>> c = [1., 1., 5.]
    >>> S = Simplex(A, b, c, which='min')
    >>> np.round(S.x, 3)
    array([0.   , 0.333, 0.667])
    >>> np.round(S.min_solution, 3)
    3.667
    """

    BOUNDED_FEASIBLE = 'Bounded Feasible'
    INFEASIBLE = 'Infeasible'
    UNBOUNDED_FEASIBLE = 'Unbounded Feasible'

    def __init__(self, A, b, c, auto=True, debug=False, which='max'):

        self._A = np.array(A)
        self._b = np.array(b)
        self._c = np.array(c)
        self._T = Tableau(A, b, c, debug=debug)

        self._iters = 0
        self.max_solution_type = None
        self.min_solution_type = None

        self.auto = auto
        self._debug = debug
        self.which = which
        self.solve()

    @property
    def A(self):
        """
        The original problem A array.

        Redefining this array will cause the Simplex's tableau to be reset (and solved if 'auto' is True).
        """
        return self._A

    @A.setter
    def A(self, value):
        """Define the problem A array and resolve if 'auto' is set."""
        self.reset()
        self._A = np.array(value)
        self.T.A = value
        if self.auto:
            self.solve()

    @property
    def b(self):
        """
        The original problem b vector such that Ax<=b for a standard maximum problem.

        Redefining this vector will cause the Simplex's tableau to be reset (and solved if 'auto' is True).
        """
        return self._b

    @b.setter
    def b(self, value):
        """Define the problem b vector and resolve if 'auto' is set."""
        self.reset()
        self._b = np.array(value)
        self.T.b = value
        if self.auto:
            self.solve()

    @property
    def c(self):
        """
        The original c vector such that c.transpose().dot(x) should be maximized in a standard maximum problem.

        Redefining this vector will cause the Simplex's tableau to be reset (and solved if 'auto' is True).
        """
        return self._c

    @c.setter
    def c(self, value):
        """Define the problem c vector and resolve if 'auto' is set."""
        self.reset()
        self._c = np.array(value)
        self.T.c = value
        if self.auto:
            self.solve()

    @property
    def debug(self):
        """The Simplex object debug boolean."""
        return self._debug

    @debug.setter
    def debug(self, value):
        """Set the Simplex object's debug flag, as well as its Tableau"""
        self._debug = value
        self.T.debug = value

    @property
    def max_solution(self):
        """
        The solution to the standard maximum problem.

        Return the solution value if the max problem is bounded feasible,
        inf if the problem is unbounded feasible, and None if the problem
        if infeasible

        Returns
        -------
        float or None
            Solution value
        """
        if not self.max_solution_type:
            self._solve_max()

        if self.max_solution_type == self.UNBOUNDED_FEASIBLE:
            return inf
        elif self.max_solution_type == self.INFEASIBLE:
            return None
        else:
            return self.c.dot(self.T.x)

    @property
    def min_solution(self):
        """
        The solution to the standard minimum problem.

        Return the solution value if the min problem is bounded feasible,
        -inf if the problem if unbounded feasible, and None if the problem
        is infeasible.

        Returns
        -------
        float or None
            Solution value or type
        """
        if not self.min_solution_type:
            self._solve_min()

        if self.min_solution_type == self.UNBOUNDED_FEASIBLE:
            return -inf
        elif self.min_solution_type == self.INFEASIBLE:
            return None
        else:
            return self.T.y.dot(self.b)

    def reset(self):
        """Reset the iteration counter, solutions, and Tableau."""
        self._iters = 0
        self.max_solution_type = None
        self.min_solution_type = None

        self._T = Tableau(self.A, self.b, self.c, debug=self.debug)

    def solve(self):
        """Solve the linear programming problem."""
        self._iters = 0

        if self.which == 'both':
            self._solve_max()
            self._solve_min()
        elif self.which == 'max':
            self._solve_max()
        else:
            self._solve_min()

    @property
    def T(self):
        """The current problem tableau."""
        return self._T

    @property
    def which(self):
        """The desired problem solution type ('max' or 'min')"""
        return self._which

    @which.setter
    def which(self, value):
        """Specify the desired solution type."""
        if value.lower() not in ['max', 'min']:
            raise ValueError("Solution flag 'which' must be 'max' or 'min'")
        self._which = value

    @property
    def x(self):
        """The tableau x vector."""
        return self.T.x

    @property
    def y(self):
        """The tableau y vector."""
        return self.T.y

    def _pivot_max_all_bi_pos(self):
        """
        Check if the problem is bounded then pivot about the optimal index.

        If all elements in the columns where -c < 0. are positive, then the
        problem is unbounded feasible. If the problem is bounded, then pivot
        about the positive A[i, j] for -c[j] < 0 which produces the smallest
        ratio b[i]/A[i,j].
        """
        neg_col_indices = np.where(-self.T.c < 0.)[0]
        if np.all(self.T.A[:, neg_col_indices] <= 0.):
            self.max_solution_type = self.UNBOUNDED_FEASIBLE

        else:
            # only pick pivots from columns where -c is negative
            mask = np.array([True] * self.T.A.size).reshape(self.T.A.shape)
            mask[:, neg_col_indices] = False
            Amask = np.ma.array(self.T.A, mask=mask)

            pp = np.ma.where(Amask > 0.)
            ratios = abs(self.T.b[pp[0]] / self.T.A[pp[0], pp[1]])
            min_index = np.where(ratios == ratios.min())[0][0]

            i0 = pp[0][min_index]
            j0 = pp[1][min_index]
            self.T.pivot(i0, j0)

    def _pivot_max_some_bi_neg(self):
        """
        Check if the problem is feasible then pivot about the optimal index.

        If all the elements in rows where b < 0 are positive, then the problem
        is infeasible. If the problem is feasible, then pivot about the negative
        A[i,j] for b[i] < 0 which produces the smallest ratio b[i]/A[i,j].
        """
        neg_row_indices = np.where(self.T.b < 0.)[0]
        if np.all(self.T.A[neg_row_indices, :] >= 0):
            self.max_solution_type = self.INFEASIBLE
        else:
            # only select pivots from rows where b is negative
            mask = np.array([True] * self.T.A.size).reshape(self.T.A.shape)
            mask[neg_row_indices, :] = False
            Amask = np.ma.array(self.T.A, mask=mask)

            pp = np.ma.where(Amask < 0.)   # possible pivots
            ratios = self.T.b[pp[0]] / self.T.A[pp[0], pp[1]]
            min_index = np.where(ratios == ratios.min())[0][0]

            i0 = pp[0][min_index]
            j0 = pp[1][min_index]
            self.T.pivot(i0, j0)

    def _pivot_min_all_negci_pos(self):
        """
        Check if the problem is bounded then pivot about the optimal point.

        If all elements in rows where b < 0 are positive, then the problem is
        unbounded feasible. If the problem is bounded, pivot about the A[i,j]
        for b[i] < 0 which produces the smallest ratio -c[j]/A[i,j].
        """
        neg_row_indices = np.where(self.T.b < 0.)[0]
        if np.all(self.T.A[neg_row_indices] >= 0.):
            self.min_solution_type = self.UNBOUNDED_FEASIBLE

        else:
            # only select pivots from rows where b is negative
            mask = np.array([True] * self.T.A.size).reshape(self.T.A.shape)
            mask[neg_row_indices, :] = False
            Amask = np.ma.array(self.T.A, mask=mask)

            pp = np.ma.where(Amask < 0.)
            ratios = abs(-self.T.c[pp[1]] / self.T.A[pp[0], pp[1]])
            min_index = np.where(ratios == min(ratios))[0][0]

            i0 = pp[0][min_index]
            j0 = pp[1][min_index]
            self.T.pivot(i0, j0)

    def _pivot_min_some_negci_neg(self):
        """
        Check if the problem is feasible then pivot about the optimal point.

        If all elements in columns where -c < 0 are negative, then the problem
        if infeasible. If the problem is feasible, pivot about the A[i,j] for
        -c[j] < 0 which produces the smallest ratio -c[j]/A[i,j].
        """
        neg_col_indices = np.where(-self.T.c < 0.)[0]
        if np.all(self.T.A[:, neg_col_indices] <= 0):
            self.min_solution_type = self.INFEASIBLE
        else:
            # only select pivots from columns where -c is negative
            mask = np.array([True] * self.T.A.size).reshape(self.T.A.shape)
            mask[:, neg_col_indices] = False
            Amask = np.ma.array(self.T.A, mask=mask)

            pp = np.ma.where(Amask > 0.)
            ratios = abs(self.T.c[pp[1]] / self.T.A[pp[0], pp[1]])
            min_index = np.where(ratios == ratios.min())[0][0]

            i0 = pp[0][min_index]
            j0 = pp[1][min_index]
            self.T.pivot(i0, j0)

    def _solve_max(self):
        """
        Find the solution to the standard maximum problem.

        Pivot until the problem's modified b and -c vectors are all >= 0, or
        until it is determined that the maximum problem is unbounded feasible
        or infeasible. A different algorithm is employed for determining the pivot
        location depending on whether or not all elements in the modified b vector
        are >= 0.
        """
        while any(self.T.b < 0.) or any(-self.T.c < 0.):
            self._iters += 1
            if self.debug:
                print("ITERATION %d:" % self._iters)

            if all(self.T.b >= 0.):
                self._pivot_max_all_bi_pos()
            else:
                self._pivot_max_some_bi_neg()

            if self.max_solution_type:
                # If the solution type is set within this loop, then the problem
                # has been determined to be unbounded feasible or infeasible
                break

        if not self.max_solution_type:
            self.max_solution_type = self.BOUNDED_FEASIBLE

    def _solve_min(self):
        """
        Find the solution to the standard minimum problem.

        Pivot until the problem's modified b and -c vectors are all >= 0, or
        until it is determined that the minimum problem is unbounded feasible
        or infeasible. A different algorithm is employed for determining the pivot
        location depending on whether or not all elements in the modified -c vector
        are >= 0.
        """
        while any(self.T.b < 0.) or any(-self.T.c < 0.):
            self._iters += 1
            if self.debug:
                print("ITERATION %d:" % self._iters)

            if all(-self.T.c >= 0.):
                self._pivot_min_all_negci_pos()
            else:
                self._pivot_min_some_negci_neg()

            if self.min_solution_type:
                # If the solution type is set within this loop, then the problem
                # has been determined to be unbounded feasible or infeasible
                break

        if not self.min_solution_type:
            self.min_solution_type = self.BOUNDED_FEASIBLE


class Tableau(object):
    """
    An object for systematically manipulating systems of equations in terms of two variable sets.

    The tableau tracks the values within a 2D array of coefficients (A), constraint vectors (b
    and c), as well as the solution vectors (r and t). The 'r' and 't' vectors generally represent
    the values of x and y which are the solution vectors to the standard maximum and minimum
    problems. Pivoting the tableau represents substitution of variables.

    Parameters
    ----------
    A: (m, n) ndarray
        Problem A array
    b: (m,) ndarray
        Problem b vector
    c: (n,) ndarray
        Problem c vector
    debug: bool, optional
        Output debug information for each solution iteration. Default=False.

    References
    ----------
    [1] Ferguson, Thomas S., Linear Programming: A Concise Introduction.

    Examples
    --------
    >>> A = [[0., 1., 2.], [-1., 0., -3.], [2., 1., 7.]]
    >>> b = [3., -2., 5.]
    >>> c = [1., 1., 5.]
    >>> T = Tableau(A, b, c)
    >>> T.pivot(2, 1)
    >>> T.A
    array([[-2., -1., -5.],
           [-1., -0., -3.],
           [ 2.,  1.,  7.]])
    """
    def __init__(self, A, b, c, debug=False):
        self._A = np.array(A)
        self._b = np.array(b)
        self._c = np.array(c)
        self._check_dimensions()

        self._init_rt_vectors()

        self.debug = debug

    @property
    def A(self):
        """The tableau A array."""
        return self._A

    @A.setter
    def A(self, value):
        """Define the tableau A array."""
        self._A = np.array(value)
        self._check_dimensions()

    @property
    def b(self):
        """The tableau b vector."""
        return self._b

    @b.setter
    def b(self, value):
        """Define the tableau b vector."""
        self._b = np.array(value)
        self._check_dimensions()

    @property
    def c(self):
        """The tableau c vector."""
        return self._c

    @c.setter
    def c(self, value):
        """Define the c vector."""
        self._c = np.array(value)
        self._check_dimensions()

    def pivot(self, row, col):
        """
        Perform a variable substitution in the system of equations about A[row, col].

        For pivot element 'p' in the A array, 'r' elements in the same row as
        the pivot, 'c' elements in the same column as the pivot, and all other
        elements denoted as 'q', the values after pivoting (denoted by '*') are
        calculated as follows:
            p* =  1 / p
            r* =  r / p
            c* = -c / p
            q* = q - rc / p

        Parameters
        ----------
        row: int
            pivot row
        col: int
            pivot column
        """
        if self.debug:
            self.print_tableau()
            print("Pivoting on row, col = %d, %d" % (row, col))
            print("A[%d, %d] = %f\n" % (row, col, self.A[row, col]))

        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()

        other_rows = [j for j in range(len(A[:, 0])) if j != row]
        other_cols = [i for i in range(len(A[0, :])) if i != col]

        self.A[row, col] = 1. / A[row, col]
        self.A[row, other_cols] =  A[row, other_cols] / A[row, col]
        self.A[other_rows, col] = -A[other_rows, col] / A[row, col]

        self.b[row] =  b[row] / A[row, col]
        self.c[col] = -c[col] / A[row, col]

        or_ = np.repeat(other_rows, len(other_cols))
        oc_ = np.tile(other_cols, len(other_rows))

        self.A[or_, oc_] = A[or_, oc_] - A[np.repeat(row, len(or_)), oc_] * \
            A[or_, np.repeat(col, len(oc_))] / np.repeat(A[row, col], len(oc_))
        self.b[other_rows] = b[other_rows] - np.repeat(b[row], len(b)-1) * A[other_rows, col] / \
            np.repeat(A[row, col], len(b)-1)
        self.c[other_cols] = c[other_cols] - np.repeat(c[col], len(c)-1) * A[row, other_cols] / \
            np.repeat(A[row, col], len(c)-1)

        self._swap_rt_values(row, col)

    def print_tableau(self):
        """Output current T information."""
        print("A'  =\n%s" % str(self.A))
        print("b'  = %s" % str(self.b.transpose()))
        print("-c' = %s" % str(-self.c))
        print("rbool = %s" % str(self.r))
        print("rdata = %s" % str(self.r.data))
        print("tbool = %s" % str(self.t))
        print("tdata = %s\n" % str(self.t.data))

    @property
    def r(self):
        """The vector which tracks the positions of the y vector which are not 0."""
        return self._r

    @property
    def t(self):
        """The vector which tracks the positions of the x vector which are not 0."""
        return self._t

    @property
    def x(self):
        """
        The x vector of the current tableau.

        The value of each element in the t vector which is not masked are the
        indices of the x vector which are not 0. The values of the x elements are
        the values of the b vector corresponding to the position of the unmasked
        t vector elements.
        """
        x = np.array([0.] * len(self.r))
        for i, ti in enumerate(self.t):
            if type(ti) is not np.ma.core.MaskedConstant:
                x[ti] = self.b[i]
        return x

    @property
    def y(self):
        """
        The y vector of the current tableau.

        The value of each element in the r vector which is not masked are the
        indices of the y vector which are not 0. The values of the y elements are
        the values of the c vector corresponding to the position of the umasked
        r vector elements.
        """
        y = np.array([0.] * len(self.t))
        for i, ri in enumerate(self.r):
            if type(ri) is not np.ma.core.MaskedConstant:
                y[ri] = -self.c[i]
        return y

    def _check_dimensions(self):
        """Verify the correct dimensions for Tableau inputs."""
        # Verify number of array dimensions
        if self.A.ndim != 2:
            raise ValueError("'A' must be a 2D array")
        if self.b.ndim != 1:
            raise ValueError("'b' must be a vector")
        if self.c.ndim != 1:
            raise ValueError("'c' must be a vector")

        # Verify matching dimension sizes
        if self.A.shape[1] != self.c.shape[0]:
            raise ValueError("'A' must have the same number of columns as elements in 'c'")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("'A' must have the same number of rows as elements in 'b'")

    def _init_rt_vectors(self):
        """Initialize r and t vectors for tracking the x and y values."""
        self._r = np.ma.array(range(len(self.c)), mask=[True] * len(self.c))
        self._t = np.ma.array(range(len(self.b)), mask=[True] * len(self.b))

    def _swap_rt_values(self, row, col):
        """
        Swap the value from r[col] with t[row] and mask/unmask if necessary.

        If the masks of the values being swapped are the same, then an x index
        is being swapped with a y index. Therefore, the masks must be switched
        to flip whether or not those indices are included in the x or y vectors.
        """
        # get original values and remove masks
        r_original = np.array(self.r.copy())
        t_original = np.array(self.t.copy())

        # swap indices
        self.r.data[col] = t_original.data[row]
        self.t.data[row] = r_original.data[col]

        # adjust mask if swapping an r and a t index
        if self.r.mask[col] == self.t.mask[row]:
            self.r.mask[col] = not self.r.mask[col]
            self.t.mask[row] = not self.t.mask[row]
