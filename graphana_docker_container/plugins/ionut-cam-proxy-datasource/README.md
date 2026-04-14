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
