package handlers

import (
	"io"
	"log/slog"
	"net/http"
	"strconv"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/job"
)

type Jobs struct {
	svc *job.Service
	log *slog.Logger
}

func NewJobs(svc *job.Service, log *slog.Logger) *Jobs { return &Jobs{svc: svc, log: log} }

// createJobRequest is the public shape of POST /jobs. It deliberately has no
// image, command or script fields: clients choose parameters, not code.
type createJobRequest struct {
	Name      string `json:"name"`
	DatasetID string `json:"dataset_id"`
	Epochs    int    `json:"epochs"`
	Workers   int    `json:"workers"`
	DataMode  string `json:"data_mode"`
	GPUs      int    `json:"gpus"`
	CPUCores  int    `json:"cpu_cores"`
	MemoryMB  int    `json:"memory_mb"`
}

func (h *Jobs) Create(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, r, h.log, err)
		return
	}
	j, err := h.svc.Create(r.Context(), job.CreateInput(req))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusCreated, j)
}

func (h *Jobs) List(w http.ResponseWriter, r *http.Request) {
	js, err := h.svc.List(r.Context())
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": js})
}

func (h *Jobs) Get(w http.ResponseWriter, r *http.Request) {
	j, err := h.svc.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, j)
}

func (h *Jobs) Start(w http.ResponseWriter, r *http.Request) {
	j, err := h.svc.Start(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, j)
}

func (h *Jobs) Stop(w http.ResponseWriter, r *http.Request) {
	j, err := h.svc.Stop(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusAccepted, j)
}

// Logs returns plain-text logs; ?tail=N limits to the last N lines.
func (h *Jobs) Logs(w http.ResponseWriter, r *http.Request) {
	tail := 500
	if v := r.URL.Query().Get("tail"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 || n > 100000 {
			writeError(w, r, h.log, apperr.Invalid("INVALID_TAIL", "tail must be an integer between 0 and 100000"))
			return
		}
		tail = n
	}
	rc, err := h.svc.Logs(r.Context(), r.PathValue("id"), tail)
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	defer rc.Close()
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = io.Copy(w, rc)
}
