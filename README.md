# Go-FastText

Golang bindings to the fasttext library.

## Usage


```go
package main

import (
        "fmt"

        fasttext "code.byted.org/zhupeihao/go-fasttext"
        "github.com/Unknwon/com"
)


var modelPath string = ''//model_path

func detect(model *fasttext.Model, str string ,k int) {
    if k==-1 {
        k = model.GetLabelNum()
    }
    
    preds, err := model.Predict(str,k)
    if err != nil {
            fmt.Println(err)
            return
    }

    fmt.Println(preds)

}

func main() {
    
    if !com.IsFile(modelPath) {
            fmt.Println("the file %s does not exist", modelPath)
            return
    }

    model := fasttext.Open(modelPath)
    defer model.Close()
    
    detect(model, "绿色守护着你", 5)
    detect(model, "you are so beautiful", -1)
}
```
