# Go-FastText

Golang bindings to the fasttext library.

## Install

```
go get -u github.com/Unknwon/com
go get -u github.com/clinjie/go-fasttext
```

## Usage


```go
package main

import (
        "fmt"

        fasttext "github.com/clinjie/go-fasttext"
        "github.com/Unknwon/com"
)


var modelPath string = ''//model_path

func main() {

    if !com.IsFile(modelPath) {
            fmt.Println("the file %s does not exist", modelPath)
            return
    }

    model := fasttext.Open(modelPath)
    defer model.Close()

    preds, _ := model.Predict("8 9", 4)
    fmt.Println(preds)
    preds, _ = model.Predict("우리 및 말에서 유래한 한국어 낱말.")
    fmt.Println(preds)
    preds, _ = model.Predict("what are you", -1)
    fmt.Println(preds)

    return
}
```
