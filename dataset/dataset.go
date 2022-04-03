package dataset

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/mkideal/mnist/mathx"
)

var (
	ErrLabel = errors.New("error label")
)

type Sample struct {
	Input *mathx.Matrix
	Label *mathx.Matrix
}

// @see http://yann.lecun.com/exdb/mnist/

func ReadTrainingSet(imageFile, labelFile string) (result []*Sample, err error) {
	if result, err = readImages(imageFile, result); err == nil {
		result, err = readLabels(labelFile, result)
	}
	return
}

func ReadTestSet(imageFile, labelFile string) (result []*Sample, err error) {
	if result, err = readImages(imageFile, result); err == nil {
		result, err = readLabels(labelFile, result)
	}
	return
}

func SplitTrainingSet(set []*Sample) (trainingdata, validationset []*Sample) {
	n := 5 * len(set) / 6
	return set[:n], set[n:]
}

func isFileExist(filename string) bool {
	_, err := os.Stat(filename)
	if err != nil {
		return !os.IsNotExist(err)
	}
	return true
}

func tryDownload(filename string) (string, error) {
	if strings.HasPrefix(filename, "http://") || strings.HasSuffix(filename, "https://") {
		cacheDir, err := os.UserHomeDir()
		if err != nil {
			cacheDir = ".cache"
		} else {
			cacheDir = filepath.Join(cacheDir, ".cache", "download")
		}
		if err := os.MkdirAll(cacheDir, 0755); err != nil {
			return "", err
		}
		_, name := path.Split(filename)
		cacheFilename := filepath.Join(cacheDir, name)
		if isFileExist(cacheFilename) {
			return cacheFilename, nil
		}

		log.Printf("downloading %s to %s", filename, cacheFilename)
		resp, err := http.Get(filename)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		out, err := os.Create(cacheFilename)
		if err != nil {
			return "", err
		}
		defer out.Close()
		if n, err := io.Copy(out, resp.Body); err != nil {
			return "", err
		} else {
			log.Printf("%s downloaded, total %d bytes", filename, n)
		}
		filename = cacheFilename
	}
	return filename, nil
}

func readImages(filename string, result []*Sample) ([]*Sample, error) {
	filename, err := tryDownload(filename)
	if err != nil {
		return result, err
	}

	file, err := os.Open(filename)
	if err != nil {
		return result, err
	}
	defer file.Close()

	gzreader, err := gzip.NewReader(file)
	if err != nil {
		return result, err
	}
	defer gzreader.Close()

	reader := bufio.NewReader(gzreader)
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
		result = make([]*Sample, num)
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
			result[i] = new(Sample)
		}
		result[i].Input = vec
	}
	return result, nil
}

func readLabels(filename string, result []*Sample) ([]*Sample, error) {
	filename, err := tryDownload(filename)
	if err != nil {
		return result, err
	}
	file, err := os.Open(filename)
	if err != nil {
		return result, err
	}

	gzreader, err := gzip.NewReader(file)
	if err != nil {
		return result, err
	}
	defer gzreader.Close()

	reader := bufio.NewReader(gzreader)
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
		result = make([]*Sample, num)
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
			result[i] = new(Sample)
		}
		result[i].Label = vec
	}
	return result, nil
}

func readInteger(reader io.Reader) (value int32, err error) {
	err = binary.Read(reader, binary.BigEndian, &value)
	return
}
