package mathx

import (
	"math"
	"math/rand"
)

type UnaryFunction func(Float) Float

func (f UnaryFunction) Add(f2 UnaryFunction) UnaryFunction {
	return func(x Float) Float {
		return f(x) + f2(x)
	}
}

func (f UnaryFunction) Sub(f2 UnaryFunction) UnaryFunction {
	return func(x Float) Float {
		return f(x) - f2(x)
	}
}

func (f UnaryFunction) Mul(f2 UnaryFunction) UnaryFunction {
	return func(x Float) Float {
		return f(x) * f2(x)
	}
}

func (f UnaryFunction) Div(f2 UnaryFunction) UnaryFunction {
	return func(x Float) Float {
		return f(x) / f2(x)
	}
}

// Rand returns a random Float which in range [0, 1)
func Rand() Float {
	return Float(rand.Intn(1000000)) / 1000000
}

func Shuffle(vec []int) {
	for i := len(vec) - 1; i >= 0; i-- {
		index := rand.Intn(i + 1)
		vec[i], vec[index] = vec[index], vec[i]
	}
}

func Constant(c Float) UnaryFunction      { return func(x Float) Float { return c } }
func KSigmoid(k Float) UnaryFunction      { return func(x Float) Float { return Sigmoid(k * x) } }
func KSigmoidPrime(k Float) UnaryFunction { return func(x Float) Float { return SigmoidPrime(k*x) * k } }

var (
	ConstantOne UnaryFunction = func(x Float) Float { return 1 }
	Identity    UnaryFunction = func(x Float) Float { return x }
	Square      UnaryFunction = func(x Float) Float { return x * x }
	Abs         UnaryFunction = func(x Float) Float { return Float(math.Abs(float64(x))) }
	Sign        UnaryFunction = func(x Float) Float {
		if x > 0 {
			return 1
		}
		return -1
	}
	Sigmoid      UnaryFunction = func(x Float) Float { return Float(1.0 / (1.0 + math.Exp(-float64(x)))) }
	SigmoidPrime UnaryFunction = func(x Float) Float {
		x = Sigmoid(x)
		return x * (1 - x)
	}
)
