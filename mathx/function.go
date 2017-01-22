package mathx

import (
	"math"
	"math/rand"
)

type UnaryFunction func(Float) Float

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

func Identity(x Float) Float { return x }
func Square(x Float) Float   { return x * x }
func Sigmoid(x Float) Float  { return Float(1.0 / (1.0 + math.Exp(-float64(x)))) }
func SigmoidPrime(x Float) Float {
	x = Sigmoid(x)
	return x * (1 - x)
}
