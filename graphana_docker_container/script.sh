#!/usr/bin/env bash
set -euo pipefail

PLUGIN_ID="ionut-cam-proxy-datasource"
PLUGIN_NAME="Ionut Cam Proxy Datasource"
EXECUTABLE="gpx_ionut_cam_proxy_datasource"
TARGET_DIR="${1:-./ionut-cam-proxy-datasource}"

mkdir -p "$TARGET_DIR/pkg/plugin" "$TARGET_DIR/src" "$TARGET_DIR/img"

cat > "$TARGET_DIR/plugin.json" <<JSON
{
  "\$schema": "https://grafana.com/schemas/plugin/v2.json",
  "type": "app",
  "id": "$PLUGIN_ID",
  "name": "$PLUGIN_NAME",
  "info": {
    "description": "Backend app plugin that proxies ESP32-CAM snapshot/stream resources through Grafana",
    "author": { "name": "Ionut / ChatGPT" },
    "keywords": ["camera", "proxy", "grafana", "esp32-cam"],
    "version": "0.0.1",
    "logos": { "small": "img/logo.svg", "large": "img/logo.svg" }
  },
  "dependencies": {
    "grafanaDependency": ">=10.0.0",
    "plugins": []
  },
  "backend": true,
  "executable": "$EXECUTABLE"
}
JSON

cat > "$TARGET_DIR/pkg/go.mod" <<'GOMOD'
module ionut-cam-proxy-datasource

go 1.22

require github.com/grafana/grafana-plugin-sdk-go v0.279.0
GOMOD

cat > "$TARGET_DIR/pkg/main.go" <<'GO'
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
GO

cat > "$TARGET_DIR/pkg/plugin/plugin.go" <<'GO'
package plugin

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
	"github.com/grafana/grafana-plugin-sdk-go/backend/app"
	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
	"github.com/grafana/grafana-plugin-sdk-go/backend/resource/httpadapter"
)

type CamProxyApp struct {
	resourceHandler backend.CallResourceHandler
	httpClient      *http.Client
	baseURL         string
}

func NewApp(_ backend.AppInstanceSettings) (app.Instance, error) {
	baseURL := strings.TrimRight(envOr("CAM_PROXY_BASE_URL", "http://127.0.0.1:8000"), "/")

	p := &CamProxyApp{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/snapshot", p.handleSnapshot)
	mux.HandleFunc("/stream", p.handleStream)
	mux.HandleFunc("/status", p.handleStatus)
	mux.HandleFunc("/healthz", p.handleHealthz)
	p.resourceHandler = httpadapter.New(mux)

	log.DefaultLogger.Info("cam proxy plugin started", "baseURL", baseURL)
	return p, nil
}

func (p *CamProxyApp) Dispose() {}

func (p *CamProxyApp) CallResource(ctx context.Context, req *backend.CallResourceRequest, sender backend.CallResourceResponseSender) error {
	return p.resourceHandler.CallResource(ctx, req, sender)
}

func (p *CamProxyApp) handleHealthz(rw http.ResponseWriter, req *http.Request) {
	rw.Header().Set("Content-Type", "application/json")
	rw.WriteHeader(http.StatusOK)
	_, _ = rw.Write([]byte(`{"ok":true}`))
}

func (p *CamProxyApp) handleStatus(rw http.ResponseWriter, req *http.Request) {
	p.proxySimple(rw, req, "/status", "text/plain; charset=utf-8")
}

func (p *CamProxyApp) handleSnapshot(rw http.ResponseWriter, req *http.Request) {
	p.proxySimple(rw, req, "/snapshot", "image/jpeg")
}

func (p *CamProxyApp) handleStream(rw http.ResponseWriter, req *http.Request) {
	target := p.baseURL + "/stream"
	reqUp, err := http.NewRequestWithContext(req.Context(), http.MethodGet, target, nil)
	if err != nil {
		http.Error(rw, fmt.Sprintf("request build failed: %v", err), http.StatusInternalServerError)
		return
	}

	resp, err := p.httpClient.Do(reqUp)
	if err != nil {
		http.Error(rw, fmt.Sprintf("upstream request failed: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, vals := range resp.Header {
		lk := strings.ToLower(k)
		if lk == "connection" || lk == "transfer-encoding" || lk == "keep-alive" || lk == "proxy-authenticate" || lk == "proxy-authorization" || lk == "te" || lk == "trailers" || lk == "upgrade" {
			continue
		}
		for _, v := range vals {
			rw.Header().Add(k, v)
		}
	}
	if rw.Header().Get("Content-Type") == "" {
		rw.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	}
	rw.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
	rw.WriteHeader(resp.StatusCode)

	flusher, _ := rw.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := rw.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if err != io.EOF {
				log.DefaultLogger.Warn("stream proxy ended", "err", err)
			}
			return
		}
	}
}

func (p *CamProxyApp) proxySimple(rw http.ResponseWriter, req *http.Request, path string, fallbackCT string) {
	target := p.baseURL + path
	if req.URL.RawQuery != "" {
		target += "?" + req.URL.RawQuery
	}
	if _, err := url.Parse(target); err != nil {
		http.Error(rw, fmt.Sprintf("invalid upstream URL: %v", err), http.StatusInternalServerError)
		return
	}

	reqUp, err := http.NewRequestWithContext(req.Context(), http.MethodGet, target, nil)
	if err != nil {
		http.Error(rw, fmt.Sprintf("request build failed: %v", err), http.StatusInternalServerError)
		return
	}
	resp, err := p.httpClient.Do(reqUp)
	if err != nil {
		http.Error(rw, fmt.Sprintf("upstream request failed: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(rw, fmt.Sprintf("upstream read failed: %v", err), http.StatusBadGateway)
		return
	}

	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = fallbackCT
	}
	rw.Header().Set("Content-Type", ct)
	rw.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
	rw.WriteHeader(resp.StatusCode)
	_, _ = rw.Write(body)
}

func envOr(k, d string) string {
	if v := strings.TrimSpace(strings.Trim(os.Getenv(k), "\"")); v != "" {
		return v
	}
	return d
}
GO

cat > "$TARGET_DIR/src/module.ts" <<'TS'
import { AppPlugin } from '@grafana/data';

export const plugin = new AppPlugin();
TS

cat > "$TARGET_DIR/package.json" <<'JSON'
{
  "name": "ionut-cam-proxy-datasource",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "build": "echo frontend placeholder"
  }
}
JSON

cat > "$TARGET_DIR/img/logo.svg" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
  <rect x="8" y="16" width="40" height="28" rx="6" stroke="currentColor" stroke-width="4"/>
  <circle cx="28" cy="30" r="9" stroke="currentColor" stroke-width="4"/>
  <path d="M48 24l8-4v20l-8-4" stroke="currentColor" stroke-width="4" stroke-linejoin="round"/>
</svg>
SVG

cat > "$TARGET_DIR/README.md" <<'MD'
# Ionut Cam Proxy Datasource

## What this plugin does

This is a minimal Grafana **backend app plugin** that proxies your private FastAPI camera endpoints through Grafana.

Provided routes:

- `/api/plugins/ionut-cam-proxy-datasource/resources/healthz`
- `/api/plugins/ionut-cam-proxy-datasource/resources/status`
- `/api/plugins/ionut-cam-proxy-datasource/resources/snapshot`
- `/api/plugins/ionut-cam-proxy-datasource/resources/stream`

The upstream base URL is read from `CAM_PROXY_BASE_URL`, defaulting to `http://127.0.0.1:8000`.

## Build the backend binary

```bash
cd pkg
go mod tidy
go build -o ../gpx_ionut_cam_proxy_datasource .
```

## Example Grafana HTML Graphics snippet

```html
<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#111;overflow:hidden;">
  <img
    src="/api/plugins/ionut-cam-proxy-datasource/resources/stream"
    alt="ESP32-CAM live feed"
    style="display:block;max-width:100%;max-height:100%;width:100%;height:auto;object-fit:contain;border-radius:8px;"
  />
</div>
```
MD

cat > "$TARGET_DIR/docker-compose.snippet.yml" <<'YAML'
services:
  grafana:
    image: grafana/grafana:latest
    network_mode: host
    container_name: grafana
    restart: unless-stopped
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./plugins:/var/lib/grafana/plugins
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=ionutqwerty
      - GF_SERVER_DOMAIN=localhost
      - GF_USERS_DEFAULT_THEME=light
      - GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=ionut-cam-proxy-datasource
      - GF_LOG_LEVEL=debug
      - CAM_PROXY_BASE_URL=http://127.0.0.1:8000

volumes:
  grafana-storage: {}
YAML

echo "Created plugin scaffold in: $TARGET_DIR"
echo
echo "Next steps:"
echo "1) cd $TARGET_DIR/pkg"
echo "2) go mod tidy"
echo "3) go build -o ../$EXECUTABLE ."
echo "4) mount $TARGET_DIR into Grafana as /var/lib/grafana/plugins/$PLUGIN_ID"
echo "5) set GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=$PLUGIN_ID"
echo "6) set CAM_PROXY_BASE_URL=http://127.0.0.1:8000"
echo
echo "Then use this in Grafana HTML Graphics:"
echo '  <img src="/api/plugins/ionut-cam-proxy-datasource/resources/stream" ... >'