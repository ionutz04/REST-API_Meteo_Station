package main

import (
	"os"

	"github.com/grafana/grafana-plugin-sdk-go/backend/app"
	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
	plugin "ionut-cam-proxy-datasource/pkg/plugin"
)

func main() {
	if err := app.Manage("ionut-cam-proxy-datasource", plugin.NewApp, app.ManageOpts{}); err != nil {
		log.DefaultLogger.Error(err.Error())
		os.Exit(1)
	}
}
