package mathx

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatrixMeta(t *testing.T) {
	mat23 := NewMatrix(2, 3)
	assert.Equal(t, 2, mat23.RowCount())
	assert.Equal(t, 3, mat23.ColCount())
	assert.Equal(t, 6, mat23.Size())

	mat23.Set(0, 1, 1)
	assert.Equal(t, Float(1), mat23.Get(0, 1))
	t.Logf("mat23: %v", *mat23)

	mat32 := mat23.Clone().T()
	assert.Equal(t, 3, mat32.RowCount())
	assert.Equal(t, 2, mat32.ColCount())
	assert.Equal(t, 6, mat32.Size())
	assert.Equal(t, Float(1), mat32.Get(1, 0))
	t.Logf("mat32: %v", *mat32)
}

func TestMatrixCalc(t *testing.T) {
	mat12 := NewMatrixOne(1, 2)
	mat12.ScaleWith(2)
	assert.Equal(t, Float(2), mat12.Get(0, 0))
	assert.Equal(t, Float(2), mat12.Get(0, 1))

	mat23 := NewMatrixOne(2, 3)
	mat23.ScaleWith(3)
	mat13 := mat12.Mul(mat23)
	assert.Equal(t, Float(12), mat13.Get(0, 0))
	assert.Equal(t, Float(12), mat13.Get(0, 1))
	assert.Equal(t, Float(12), mat13.Get(0, 2))

	mat12_1 := NewMatrix(1, 2)
	mat12_1.Set(0, 0, 1)
	mat12_1.Set(0, 1, 2)
	mat12_1.AddWith(mat12)
	assert.Equal(t, Float(3), mat12_1.Get(0, 0))
	assert.Equal(t, Float(4), mat12_1.Get(0, 1))

	mat12_2 := NewMatrix(1, 2)
	mat12_2.Set(0, 0, 3)
	mat12_2.Set(0, 1, 2)
	mat12_2.SubWith(mat12)
	assert.Equal(t, Float(1), mat12_2.Get(0, 0))
	assert.Equal(t, Float(0), mat12_2.Get(0, 1))

	t.Logf("mat12: %v", *mat12)
	t.Logf("mat12_1: %v", *mat12_1)
	t.Logf("mat12_2: %v", *mat12_2)
	t.Logf("mat23: %v", *mat23)
}

func TestMatrixNorm(t *testing.T) {
	mat := NewMatrixOne(2, 3).ScaleWith(3)
	assert.Equal(t, Float(18), mat.L1())
	assert.Equal(t, Float(math.Sqrt(54)), mat.L2())
	assert.Equal(t, Float(math.Sqrt(54)), mat.Lp(2))
	assert.Equal(t, Float(math.Pow(162, 1.0/3)), mat.Lp(3))
}

func TestMatrixString(t *testing.T) {
	mat00 := NewMatrix(0, 0)
	assert.Equal(t, "[]", mat00.String())
	mat11 := NewMatrix(1, 1)
	assert.Equal(t, "[[0.000000]]", mat11.String())
	mat12 := NewMatrix(1, 2)
	assert.Equal(t, "[[0.000000 0.000000]]", mat12.String())
	mat23 := NewMatrixOne(2, 3).ScaleWith(3)
	assert.Equal(t, "[[3.000000 3.000000 3.000000] [3.000000 3.000000 3.000000]]", mat23.String())
	assert.Equal(t, "[[3.000000 3.000000] [3.000000 3.000000] [3.000000 3.000000]]", mat23.T().String())
}

func TestMatrixMap(t *testing.T) {
	mat := NewMatrix(2, 3)
	mat = mat.Map(func(x Float) Float { return 1 })
	assert.True(t, NewMatrixOne(2, 3).Equal(mat))
}
