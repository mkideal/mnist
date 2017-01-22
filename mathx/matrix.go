package mathx

import (
	"bytes"
	"fmt"
	"math"
)

const precision = 1E-6

type Matrix struct {
	m, n      int
	transpose bool
	data      []Float
}

func NewMatrix(m, n int) *Matrix {
	mat := new(Matrix)
	mat.m = m
	mat.n = n
	mat.data = make([]Float, m*n)
	return mat
}

func NewMatrixWithRowVector(vec []Float) *Matrix {
	mat := new(Matrix)
	mat.m = 1
	mat.n = len(vec)
	mat.data = vec
	return mat
}

func NewMatrixWithColVector(vec []Float) *Matrix {
	mat := new(Matrix)
	mat.m = len(vec)
	mat.n = 1
	mat.data = vec
	return mat
}

func NewSquareMatrix(n int) *Matrix {
	return NewMatrix(n, n)
}

func NewUnitSquareMatrix(n int) *Matrix {
	mat := new(Matrix)
	mat.m = n
	mat.n = n
	size := n * n
	mat.data = make([]Float, size)
	for i := 0; i < n; i++ {
		mat.data[i*n+i] = 1
	}
	return mat
}

func NewMatrixOne(m, n int) *Matrix {
	return NewMatrixWithValue(m, n, 1)
}

func NewMatrixWithValue(m, n int, x Float) *Matrix {
	mat := new(Matrix)
	mat.m = m
	mat.n = n
	size := m * n
	mat.data = make([]Float, size)
	for i := 0; i < size; i++ {
		mat.data[i] = x
	}
	return mat
}

func (mat *Matrix) Reset() *Matrix {
	for i, size := 0, len(mat.data); i < size; i++ {
		mat.data[i] = 0
	}
	return mat
}

func (mat *Matrix) Clone() *Matrix {
	mat2 := NewMatrix(mat.m, mat.n)
	mat2.transpose = mat.transpose
	for i, x := range mat.data {
		mat2.data[i] = x
	}
	return mat2
}

func (mat *Matrix) Equal(mat2 *Matrix) bool {
	m, n := mat.RowCount(), mat.ColCount()
	if m != mat2.RowCount() || n != mat2.ColCount() {
		return false
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			delta := math.Abs(float64(mat.Get(i, j) - mat2.Get(i, j)))
			if delta > precision {
				return false
			}
		}
	}
	return true
}

func (mat *Matrix) RowCount() int {
	if mat.transpose {
		return mat.n
	}
	return mat.m
}

func (mat *Matrix) ColCount() int {
	if mat.transpose {
		return mat.m
	}
	return mat.n
}

func (mat *Matrix) Size() int { return mat.m * mat.n }

func (mat *Matrix) getPtr(i, j int) *Float {
	if mat.transpose {
		return &mat.data[j*mat.n+i]
	}
	return &mat.data[i*mat.n+j]
}

func (mat *Matrix) Get(i, j int) Float {
	return *mat.getPtr(i, j)
}

func (mat *Matrix) Set(i, j int, x Float) *Matrix {
	*mat.getPtr(i, j) = x
	return mat
}

func (mat *Matrix) T() *Matrix {
	mat2 := mat.Clone()
	mat2.transpose = !mat2.transpose
	return mat2
}

func (mat *Matrix) SelfT() *Matrix {
	mat.transpose = !mat.transpose
	return mat
}

func (mat *Matrix) Add(mat2 *Matrix) *Matrix {
	return mat.addTo(mat2, NewMatrix(mat.RowCount(), mat.ColCount()))
}

func (mat *Matrix) AddWith(mat2 *Matrix) *Matrix {
	return mat.addTo(mat2, mat)
}

func (mat *Matrix) addTo(mat2, ans *Matrix) *Matrix {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans.Set(i, j, mat.Get(i, j)+mat2.Get(i, j))
		}
	}
	return ans
}

func (mat *Matrix) Sub(mat2 *Matrix) *Matrix {
	return mat.subTo(mat2, NewMatrix(mat.RowCount(), mat.ColCount()))
}

func (mat *Matrix) SubWith(mat2 *Matrix) *Matrix {
	return mat.subTo(mat2, mat)
}

func (mat *Matrix) subTo(mat2, ans *Matrix) *Matrix {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans.Set(i, j, mat.Get(i, j)-mat2.Get(i, j))
		}
	}
	return ans
}

func (mat *Matrix) HadamardProduct(mat2 *Matrix) *Matrix {
	return mat.hadamardProductTo(mat2, NewMatrix(mat.RowCount(), mat.ColCount()))
}

func (mat *Matrix) HadamardProductWith(mat2 *Matrix) *Matrix {
	return mat.hadamardProductTo(mat2, mat)
}

func (mat *Matrix) hadamardProductTo(mat2, ans *Matrix) *Matrix {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans.Set(i, j, mat.Get(i, j)*mat2.Get(i, j))
		}
	}
	return ans
}

func (mat *Matrix) Map(mapfunc UnaryFunction) *Matrix {
	return mat.mapTo(mapfunc, NewMatrix(mat.RowCount(), mat.ColCount()))
}

func (mat *Matrix) MapWith(mapfunc UnaryFunction) *Matrix {
	return mat.mapTo(mapfunc, mat)
}

func (mat *Matrix) mapTo(mapfunc UnaryFunction, ans *Matrix) *Matrix {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans.Set(i, j, mapfunc(mat.Get(i, j)))
		}
	}
	return ans
}

func (mat *Matrix) Scale(v Float) *Matrix {
	return mat.scaleTo(v, NewMatrix(mat.RowCount(), mat.ColCount()))
}

func (mat *Matrix) ScaleWith(v Float) *Matrix {
	return mat.scaleTo(v, mat)
}

func (mat *Matrix) scaleTo(v Float, ans *Matrix) *Matrix {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans.Set(i, j, mat.Get(i, j)*v)
		}
	}
	return ans
}

func (mat *Matrix) Mul(right *Matrix) *Matrix {
	m, n := mat.RowCount(), right.ColCount()
	l := mat.ColCount()
	if l != right.RowCount() {
		panic(fmt.Sprintf("Matrix.Mul: dim mismatch: %dx%d vs %dx%d", m, l, right.RowCount(), n))
	}
	ans := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var tmp Float
			for k := 0; k < l; k++ {
				tmp += mat.Get(i, k) * right.Get(k, j)
			}
			ans.Set(i, j, tmp)
		}
	}
	return ans
}

func (mat *Matrix) Accumulate(mapfunc UnaryFunction) Float {
	if mapfunc == nil {
		mapfunc = Identity
	}
	var ans Float
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ans += mapfunc(mat.Get(i, j))
		}
	}
	return ans
}

// L1 norm
func (mat *Matrix) L1() Float {
	return mat.Accumulate(func(x Float) Float {
		return Float(math.Abs(float64(x)))
	})
}

// L2 norm
func (mat *Matrix) L2() Float {
	return Float(math.Sqrt(float64(mat.Accumulate(func(x Float) Float {
		return Float(x * x)
	}))))
}

// Lp norm(p >= 1)
func (mat *Matrix) Lp(p Float) Float {
	return Float(math.Pow(float64(mat.Accumulate(func(x Float) Float {
		return Float(math.Pow(float64(x), float64(p)))
	})), 1.0/float64(p)))
}

func (mat Matrix) String() string {
	var buf bytes.Buffer
	buf.WriteByte('[')
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		if i > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteByte('[')
		for j := 0; j < n; j++ {
			if j > 0 {
				buf.WriteByte(' ')
			}
			fmt.Fprintf(&buf, "%.6f", mat.Get(i, j))
		}
		buf.WriteByte(']')
	}
	buf.WriteByte(']')
	return buf.String()
}

func (mat *Matrix) RandInit(min, max Float) *Matrix {
	for i, size := 0, len(mat.data); i < size; i++ {
		mat.data[i] = Rand()*(max-min) + min
	}
	return mat
}

func (mat *Matrix) Slice() []Float {
	return mat.data
}

func (mat *Matrix) MinElem() (row, col int, value Float) {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			x := mat.Get(i, j)
			if (i == 0 && j == 0) || x < value {
				row, col, value = i, j, x
			}
		}
	}
	return
}

func (mat *Matrix) MaxElem() (row, col int, value Float) {
	m, n := mat.RowCount(), mat.ColCount()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			x := mat.Get(i, j)
			if (i == 0 && j == 0) || x > value {
				row, col, value = i, j, x
			}
		}
	}
	return
}
