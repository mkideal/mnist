package dataloader

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"os"

	"github.com/mkideal/mnist/mathx"
)

var (
	ErrLabel = errors.New("error label")
)

type Data struct {
	Input  *mathx.Matrix
	Output *mathx.Matrix
}

// @see http://yann.lecun.com/exdb/mnist/

func ReadTrainingSet(imageFile, labelFile string) (result []*Data, err error) {
	if result, err = readImages(imageFile, result); err == nil {
		result, err = readLabels(labelFile, result)
	}
	return
}

func ReadTestSet(imageFile, labelFile string) (result []*Data, err error) {
	if result, err = readImages(imageFile, result); err == nil {
		result, err = readLabels(labelFile, result)
	}
	return
}

func SplitTrainingSet(set []*Data) (trainingdata, validationset []*Data) {
	n := 5 * len(set) / 6
	return set[:n], set[n:]
}

func readImages(filename string, result []*Data) ([]*Data, error) {
	file, err := os.Open(filename)
	if err != nil {
		return result, err
	}
	reader := bufio.NewReader(file)
	// read magic number
	if _, err = readInteger(reader); err != nil {
		return result, err
	}

	// read number of items
	num, err := readInteger(reader)
	if err != nil {
		return result, err
	}

	// read rowSize
	rowSize, err := readInteger(reader)
	if err != nil {
		return result, err
	}
	// read colSize
	colSize, err := readInteger(reader)
	if err != nil {
		return result, err
	}

	if len(result) == 0 {
		result = make([]*Data, num)
	}

	// read items
	var b byte
	for i := int32(0); i < num; i++ {
		vec := mathx.NewMatrix(int(rowSize*colSize), 1)
		for j := int32(0); j < rowSize; j++ {
			for k := int32(0); k < colSize; k++ {
				b, err = reader.ReadByte()
				if err != nil {
					return result, err
				}
				vec.Set(int(j*colSize+k), 0, mathx.Float(b)/255)
			}
		}
		if result[i] == nil {
			result[i] = new(Data)
		}
		result[i].Input = vec
	}
	return result, nil
}

func readLabels(filename string, result []*Data) ([]*Data, error) {
	file, err := os.Open(filename)
	if err != nil {
		return result, err
	}
	reader := bufio.NewReader(file)
	// read magic number
	if _, err := readInteger(reader); err != nil {
		return result, err
	}

	// read number of items
	num, err := readInteger(reader)
	if err != nil {
		return result, err
	}

	if len(result) == 0 {
		result = make([]*Data, num)
	}

	// read items
	for i := int32(0); i < num; i++ {
		b, err := reader.ReadByte()
		if err != nil {
			return result, err
		}
		if b < 0 || b > 9 {
			return result, ErrLabel
		}
		vec := mathx.NewMatrix(10, 1)
		vec.Set(int(b), 0, 1)
		if result[i] == nil {
			result[i] = new(Data)
		}
		result[i].Output = vec
	}
	return result, nil
}

func readInteger(reader io.Reader) (value int32, err error) {
	err = binary.Read(reader, binary.BigEndian, &value)
	return
}
