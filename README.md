Learn how to recognize handwriting digits from scratch
======================================================

## Dataset

Download dataset [MNIST](http://yann.lecun.com/exdb/mnist/) and unpack it to any position, e.g. `/path/to/mnist/dataset`

## Build and Run

```sh
go build
./mnist -d /path/to/mnist/dataset
```

If `-d` not specified, path `./dataset` will be used.

## Example output

	epoch  1: accuracy = 93.59%
	epoch  2: accuracy = 94.30%
	epoch  3: accuracy = 94.74%
	epoch  4: accuracy = 95.16%
	epoch  5: accuracy = 95.23%
	epoch  6: accuracy = 95.70%
	epoch  7: accuracy = 95.56%
	epoch  8: accuracy = 95.78%
	epoch  9: accuracy = 95.82%
	epoch 10: accuracy = 95.70%
