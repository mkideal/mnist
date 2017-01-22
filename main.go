package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mkideal/minist/dataloader"
	"github.com/mkideal/minist/mathx"
)

const (
	trainingImageFile = "./dataset/train-images-idx3-ubyte"
	trainingLabelFile = "./dataset/train-labels-idx1-ubyte"
	testImageFile     = "./dataset/t10k-images-idx3-ubyte"
	testLabelFile     = "./dataset/t10k-labels-idx1-ubyte"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// 创建网络: 28*[x1*...xk*]10
	net := NewNetwork([]int{28 * 28, 24, 10})

	// 读取训练数据
	trainingdata, err := dataloader.ReadTrainingSet(trainingImageFile, trainingLabelFile)
	if err != nil {
		panic(err)
	}
	trainingdata, _ = dataloader.SplitTrainingSet(trainingdata)

	// 读取测试数据
	testdata, err := dataloader.ReadTestSet(testImageFile, testLabelFile)
	if err != nil {
		panic(err)
	}

	// 使用随机梯度下降法执行训练
	net.train(trainingdata, testdata, 3)
}

type Network struct {
	weights      []*mathx.Matrix
	biases       []*mathx.Matrix
	nablaWeights []*mathx.Matrix
	nablaBiases  []*mathx.Matrix
}

func NewNetwork(numNodes []int) *Network {
	net := new(Network)
	n := len(numNodes) - 1
	net.weights = make([]*mathx.Matrix, n)
	net.biases = make([]*mathx.Matrix, n)
	net.nablaWeights = make([]*mathx.Matrix, n)
	net.nablaBiases = make([]*mathx.Matrix, n)
	for i := 0; i < n; i++ {
		net.weights[i] = mathx.NewMatrix(numNodes[i+1], numNodes[i]).RandInit(-0.001, 0.001)
		net.biases[i] = mathx.NewMatrix(numNodes[i+1], 1).RandInit(-0.001, 0.001)
		net.nablaWeights[i] = mathx.NewMatrix(net.weights[i].RowCount(), net.weights[i].ColCount())
		net.nablaBiases[i] = mathx.NewMatrix(net.biases[i].RowCount(), 1)
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
			fmt.Printf("epoch %d: error precision: %.2f%%\n", i+1, errorPrecision*100)
		}
	}
}

func (net *Network) updateMiniBatch(dataSet []*dataloader.Data, eta mathx.Float) {
	n := len(net.weights)
	eta /= mathx.Float(len(dataSet))
	for i := 0; i < n; i++ {
		net.nablaWeights[i].Reset()
		net.nablaBiases[i].Reset()
	}
	deltaNablaWeights := make([]*mathx.Matrix, n)
	deltaNablaBiases := make([]*mathx.Matrix, n)
	for i := 0; i < n; i++ {
		deltaNablaWeights[i] = mathx.NewMatrix(net.weights[i].RowCount(), net.weights[i].ColCount())
		deltaNablaBiases[i] = mathx.NewMatrix(net.biases[i].RowCount(), 1)
	}
	for _, data := range dataSet {
		net.backprop(data, deltaNablaWeights, deltaNablaBiases)
		for i := range net.nablaWeights {
			net.nablaWeights[i].AddWith(deltaNablaWeights[i])
			net.nablaBiases[i].AddWith(deltaNablaBiases[i])
		}
	}
	for i := range net.weights {
		net.weights[i].SubWith(net.nablaWeights[i].ScaleWith(eta))
		net.biases[i].SubWith(net.nablaBiases[i].ScaleWith(eta))
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
		act = z.Map(mathx.Sigmoid)
		acts = append(acts, act)
	}

	delta := net.costDerivative(acts[n], data.Output).MapMul(zs[n-1].Map(mathx.SigmoidPrime).Clone().T())

	nablaWeights[n-1] = delta.Mul(acts[n-1].Clone().T())
	nablaBiases[n-1] = delta.Clone()
	for i := n - 2; i >= 0; i-- {
		z := zs[i]
		sp := z.Map(mathx.SigmoidPrime)
		delta = net.weights[i+1].Clone().T().Mul(delta).MapMul(sp)
		nablaWeights[i] = delta.Mul(acts[i].Clone().T())
		nablaBiases[i] = delta.Clone()
	}
}

func (net *Network) costDerivative(act, output *mathx.Matrix) *mathx.Matrix {
	return act.Sub(output)
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
		if !net.test(data) {
			num++
		}
	}
	return mathx.Float(num) / mathx.Float(total)
}

func (net *Network) feedforward(input *mathx.Matrix) *mathx.Matrix {
	n := len(net.weights)
	for i := 0; i < n; i++ {
		input = net.weights[i].Mul(input).AddWith(net.biases[i]).MapWith(mathx.Sigmoid)
	}
	return input
}

func shuffle(dataSet []*dataloader.Data) {
	for i := len(dataSet) - 1; i >= 0; i-- {
		index := rand.Intn(i + 1)
		dataSet[i], dataSet[index] = dataSet[index], dataSet[i]
	}
}
