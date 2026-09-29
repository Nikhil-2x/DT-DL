package handlers

import (
	"log/slog"
	"net/http"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/http/middleware"
	"dtdl/backend/internal/job"
)

type Deps struct {
	Datasets       *dataset.Service
	Jobs           *job.Service
	MaxUploadBytes int64
	Log            *slog.Logger
}

// NewRouter wires all routes (Go 1.22+ method/path patterns) and middleware.
func NewRouter(d Deps) http.Handler {
	mux := http.NewServeMux()
	ds := NewDatasets(d.Datasets, d.MaxUploadBytes, d.Log)
	js := NewJobs(d.Jobs, d.Log)

	mux.HandleFunc("GET /api/v1/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("POST /api/v1/datasets", ds.Upload)
	mux.HandleFunc("GET /api/v1/datasets", ds.List)
	mux.HandleFunc("GET /api/v1/datasets/{id}", ds.Get)
	mux.HandleFunc("GET /api/v1/datasets/{id}/download", ds.Download)
	mux.HandleFunc("DELETE /api/v1/datasets/{id}", ds.Delete)

	mux.HandleFunc("POST /api/v1/jobs", js.Create)
	mux.HandleFunc("GET /api/v1/jobs", js.List)
	mux.HandleFunc("GET /api/v1/jobs/{id}", js.Get)
	mux.HandleFunc("POST /api/v1/jobs/{id}/start", js.Start)
	mux.HandleFunc("POST /api/v1/jobs/{id}/stop", js.Stop)
	mux.HandleFunc("GET /api/v1/jobs/{id}/logs", js.Logs)

	// Anything else (including wrong methods on known paths) gets the same
	// JSON error shape instead of the mux's plain-text 404/405.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, r, d.Log, apperr.NotFound("ROUTE_NOT_FOUND", "no such route"))
	})

	var h http.Handler = mux
	h = middleware.Recover(d.Log, h)
	h = middleware.Logging(d.Log, h)
	h = middleware.WithRequestID(h)
	return h
}
