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
