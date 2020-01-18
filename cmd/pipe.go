package cmd

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/Unknwon/com"
	fasttext "github.com/clinjie/go-fasttext"
	_ "github.com/k0kubun/pp"
	"github.com/spf13/cobra"
)

// var (
// 	unsupervisedModelPath string
// )

// predictCmd represents the predict command
var pipeCmd = &cobra.Command{
	Use:   "pipe -m [path_to_model]",
	Short: "Perform word analogy on a query using an input model",
	Run: func(cmd *cobra.Command, args []string) {
		if !com.IsFile(unsupervisedModelPath) {
			fmt.Println("the file %s does not exist", unsupervisedModelPath)
			return
		}

		// create a model object
		model := fasttext.Open(unsupervisedModelPath)
		// close the model at the end
		defer model.Close()

		// Receive a message from stdin with long polling enabled.
		c := make(chan os.Signal)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		go func() {
			<-c
			close(c)
			os.Exit(0)
		}()

		// Input
		in := make(chan string, 1024)
		go pollIn(in, args)

		// Output
		out := make(chan string, 1024)
		go pollOut(out)
		// perform the prediction

		no := os.Getenv("GOMAXPROCS")
		var wg sync.WaitGroup
		for i := 0; i < no; i++ {
			wg.Add(1)
			go process(i, in, out, model, &wg)
		}
		wg.Wait()
	},
}

func process(i int, in <-chan string, out chan<- string, model *fasttext.Model, wg *sync.WaitGroup) {
	for a := range in {
		wordvec, err := model.Wordvec(a)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error at %s\n", a)
		}
		out <- fmt.Sprintf("%+v", wordvec)
	}
	wg.Done()
}

func pollOut(out <-chan string) {
	for o := range out {
		fmt.Printf("%s\n", o)
	}
}

func pollIn(in chan<- string, args []string) {
	if len(args) == 0 {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			in <- scanner.Text()
		}
	} else {
		for _, a := range args {
			in <- a
		}
	}
	close(in)
}

func init() {
	pipeCmd.Flags().StringVarP(&unsupervisedModelPath, "model", "m", "", "path to the fasttext model")
	rootCmd.AddCommand(pipeCmd)
}
