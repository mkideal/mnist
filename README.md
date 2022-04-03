Learn how to recognize handwriting digits from scratch
======================================================

## Install

```sh
go get github.com/mkideal/mnist
```

## Dataset

Download dataset [MNIST](http://yann.lecun.com/exdb/mnist/) and unpack it to any position, e.g. `/path/to/mnist/dataset`

## Run

```sh
mnist
```

or

```sh
mnist -d /path/to/mnist/dataset
```

If `-d` not specified, path `./dataset` will be used.

## Example output: %x is accuracy

	epoch  1: 93.72%
	epoch  2: 94.35%
	epoch  3: 95.02%
	epoch  4: 95.44%
	epoch  5: 95.38%
	epoch  6: 95.53%
	epoch  7: 95.50%
	epoch  8: 94.83%
	epoch  9: 95.94%
	epoch 10: 95.99%
	epoch 11: 95.88%
	epoch 12: 96.31%
	epoch 13: 96.29%
	epoch 14: 96.22%
	epoch 15: 96.31%
	epoch 16: 96.01%
	epoch 17: 96.24%
	epoch 18: 96.34%
	epoch 19: 96.12%
	epoch 20: 96.09%
