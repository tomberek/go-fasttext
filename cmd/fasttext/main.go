package main

import (
	"bufio"
	// "encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/ekzhu/go-fasttext"
	"github.com/jdkato/prose/v2"
	_ "github.com/mattn/go-sqlite3"
)

func main() {
	dbpath := os.Getenv("FASTTEXT_DB_PATH")
	vecpath := os.Getenv("FASTTEXT_VEC_PATH")
	ft := fasttext.NewFastText(dbpath)
	vecFile, err := os.OpenFile(vecpath, os.O_RDONLY, 0644)
	if err != nil {
		fmt.Println(err)
	}
	defer vecFile.Close()
	err = ft.BuildDB(vecFile)

	runner := func(line string) {
		var embs [fasttext.Dim]float64
		defer func() {
			recover()
			// output, err := json.Marshal(embs)
			// if err != nil {
			// 	panic(err)
			// }
			// fmt.Println(string(output))
			for _, item := range embs {
				fmt.Printf("%f ", item)
			}
			fmt.Printf("\n")
		}()
		fd, err := os.OpenFile(line, os.O_RDONLY, 0644)
		if err != nil {
			panic(err)
		}
		var data = make([]byte, 1024*48)
		n, err := io.ReadFull(fd, data)
		if err != nil && err != io.ErrUnexpectedEOF {
			panic(err)
		}
		doc, err := prose.NewDocument(string(data[0:n]),
			prose.WithExtraction(false),
			prose.WithSegmentation(false),
			prose.WithTagging(false),
			prose.WithTokenization(true),
		)
		if err == nil {
			outCh := make(chan []float64)
			var wg sync.WaitGroup
			numWorkers := 32
			workers := make(chan struct{}, numWorkers)
			go func(wg *sync.WaitGroup) {
				for emb := range outCh {
					for i, v := range emb {
						embs[i] += v
					}
					wg.Done()
				}
			}(&wg)
			for _, tok := range doc.Tokens() {
				workers <- struct{}{}
				wg.Add(1)
				go func(wg *sync.WaitGroup, token string) {
					emb, err := ft.GetEmb(token)
					if err != nil {
						outCh <- nil
					} else {
						outCh <- emb
					}
					<-workers
				}(&wg, tok.Text)
			}
			wg.Wait()
			close(outCh)

			itemCount := len(doc.Tokens())
			for i, _ := range embs {
				embs[i] = embs[i] / float64(itemCount)
			}

		}
		// prose.WithSegmentation(false),
		// prose.WithTagging(false),
		// prose.WithTokenization(true),

	}
	if len(os.Args) >= 2 {
		for _, line := range os.Args[1:] {
			runner(line)
		}
		return
	}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		runner(line)
	}
}
