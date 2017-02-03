package main

import (
	"flag"
	"fmt"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/mkideal/minist/dataloader"
	"github.com/mkideal/minist/mathx"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	flDatasetPath := flag.String("d", "./dataset", "minist dataset path")
	flag.Parse()

	var (
		trainingImageFile = filepath.Join(*flDatasetPath, "train-images-idx3-ubyte")
		trainingLabelFile = filepath.Join(*flDatasetPath, "train-labels-idx1-ubyte")
		testImageFile     = filepath.Join(*flDatasetPath, "t10k-images-idx3-ubyte")
		testLabelFile     = filepath.Join(*flDatasetPath, "t10k-labels-idx1-ubyte")
	)

	net := NewNetwork([]int{28 * 28, 24, 10})

	// read training data
	trainingdata, err := dataloader.ReadTrainingSet(trainingImageFile, trainingLabelFile)
	if err != nil {
		panic(err)
	}
	trainingdata, _ = dataloader.SplitTrainingSet(trainingdata)

	// read test data
	testdata, err := dataloader.ReadTestSet(testImageFile, testLabelFile)
	if err != nil {
		panic(err)
	}

	// train(and test)
	net.train(trainingdata, testdata, 4)
}

type Network struct {
	weights     []*mathx.Matrix
	biases      []*mathx.Matrix
	actfuncs    []mathx.UnaryFunction
	actderfuncs []mathx.UnaryFunction
}

func NewNetwork(numNodes []int) *Network {
	net := new(Network)
	n := len(numNodes) - 1
	net.weights = make([]*mathx.Matrix, n)
	net.biases = make([]*mathx.Matrix, n)
	net.actfuncs = make([]mathx.UnaryFunction, n)
	net.actderfuncs = make([]mathx.UnaryFunction, n)
	for i := 0; i < n; i++ {
		net.weights[i] = mathx.NewMatrix(numNodes[i+1], numNodes[i]).RandInit(-0.001, 0.001)
		net.biases[i] = mathx.NewMatrix(numNodes[i+1], 1).RandInit(-0.001, 0.001)
		if i+1 == n {
			net.actfuncs[i] = mathx.Sigmoid
			net.actderfuncs[i] = mathx.SigmoidPrime
		} else {
			net.actfuncs[i] = mathx.Sigmoid
			net.actderfuncs[i] = mathx.SigmoidPrime
		}
	}
	return net
}

func (net *Network) train(dataSet, testdata []*dataloader.Data, eta mathx.Float) {
	var (
		times         = 20
		miniBatchSize = len(dataSet) / 6000
	)
	for i := 0; i < times; i++ {
		shuffle(dataSet)
		for j := 0; j+miniBatchSize < len(dataSet); j += miniBatchSize {
			net.updateMiniBatch(dataSet[j:j+miniBatchSize], eta)
		}
		if len(testdata) > 0 {
			errorPrecision := net.evaluate(testdata)
			fmt.Printf("epoch %2d: %.2f%%\n", i+1, errorPrecision*100)
		}
	}
}

func (net *Network) updateMiniBatch(dataSet []*dataloader.Data, eta mathx.Float) {
	n := len(net.weights)
	eta /= mathx.Float(len(dataSet))
	nablaWeights := make([]*mathx.Matrix, n)
	nablaBiases := make([]*mathx.Matrix, n)
	for i := 0; i < n; i++ {
		nablaWeights[i] = mathx.NewMatrix(net.weights[i].RowCount(), net.weights[i].ColCount())
		nablaBiases[i] = mathx.NewMatrix(net.biases[i].RowCount(), 1)
	}
	deltaNablaWeights := make([]*mathx.Matrix, n)
	deltaNablaBiases := make([]*mathx.Matrix, n)
	for i := 0; i < n; i++ {
		deltaNablaWeights[i] = mathx.NewMatrix(net.weights[i].RowCount(), net.weights[i].ColCount())
		deltaNablaBiases[i] = mathx.NewMatrix(net.biases[i].RowCount(), 1)
	}
	for _, data := range dataSet {
		net.backprop(data, deltaNablaWeights, deltaNablaBiases)
		for i := range nablaWeights {
			nablaWeights[i].AddWith(deltaNablaWeights[i])
			nablaBiases[i].AddWith(deltaNablaBiases[i])
		}
	}
	for i := range net.weights {
		net.weights[i].SubWith(nablaWeights[i].ScaleWith(eta))
		net.biases[i].SubWith(nablaBiases[i].ScaleWith(eta))
	}
}

func (net *Network) backprop(data *dataloader.Data, nablaWeights, nablaBiases []*mathx.Matrix) {
	n := len(net.weights)
	for i := 0; i < n; i++ {
		nablaWeights[i].Reset()
		nablaBiases[i].Reset()
	}
	act := data.Input
	acts := []*mathx.Matrix{act}
	zs := make([]*mathx.Matrix, 0, n)
	for i := 0; i < n; i++ {
		z := net.weights[i].Mul(act).AddWith(net.biases[i])
		zs = append(zs, z)
		act = z.Map(net.actfuncs[i])
		acts = append(acts, act)
	}

	delta := net.costDerivative(acts[n], data.Output).HadamardProduct(zs[n-1].Map(net.actderfuncs[n-1]).T())

	nablaWeights[n-1] = delta.Mul(acts[n-1].T())
	nablaBiases[n-1] = delta.Clone()
	for i := n - 2; i >= 0; i-- {
		z := zs[i]
		sp := z.Map(net.actderfuncs[i])
		delta = net.weights[i+1].T().Mul(delta).HadamardProduct(sp)
		nablaWeights[i] = delta.Mul(acts[i].T())
		nablaBiases[i] = delta.Clone()
	}
}

func (net *Network) costDerivative(act, output *mathx.Matrix) *mathx.Matrix {
	return act.Sub(output).MapWith(cube)
}

func cube(x mathx.Float) mathx.Float {
	return x * x * x
}

func (net *Network) test(data *dataloader.Data) bool {
	output := net.feedforward(data.Input)
	i, _, _ := data.Output.MaxElem()
	j, _, _ := output.MaxElem()
	return i == j
}

func (net *Network) evaluate(dataSet []*dataloader.Data) mathx.Float {
	total := len(dataSet)
	if total == 0 {
		return 0
	}
	num := 0
	for _, data := range dataSet {
		if net.test(data) {
			num++
		}
	}
	return mathx.Float(num) / mathx.Float(total)
}

func (net *Network) feedforward(input *mathx.Matrix) *mathx.Matrix {
	n := len(net.weights)
	for i := 0; i < n; i++ {
		input = net.weights[i].Mul(input).AddWith(net.biases[i]).MapWith(net.actfuncs[i])
	}
	return input
}

func shuffle(dataSet []*dataloader.Data) {
	for i := len(dataSet) - 1; i >= 0; i-- {
		index := rand.Intn(i + 1)
		dataSet[i], dataSet[index] = dataSet[index], dataSet[i]
	}
}
