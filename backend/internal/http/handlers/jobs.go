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
// image or command fields: clients choose parameters (and may reference an
// uploaded project script by id), never an arbitrary command or image.
type createJobRequest struct {
	Name      string `json:"name"`
	DatasetID string `json:"dataset_id"`
	ScriptID  string `json:"script_id"`
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

// Artifacts lists the checkpoints a job has saved.
func (h *Jobs) Artifacts(w http.ResponseWriter, r *http.Request) {
	list, err := h.svc.Artifacts(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	resp := map[string]any{"artifacts": list}
	if len(list) > 0 {
		resp["latest_epoch"] = list[len(list)-1].Epoch
	}
	writeJSON(w, http.StatusOK, resp)
}

// Model downloads a checkpoint file (the latest, or ?epoch=N). The file is a
// PyTorch checkpoint: {"model_state", "optimizer_state", "epoch"}.
func (h *Jobs) Model(w http.ResponseWriter, r *http.Request) {
	epoch := -1
	if v := r.URL.Query().Get("epoch"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			writeError(w, r, h.log, apperr.Invalid("INVALID_EPOCH", "epoch must be a non-negative integer"))
			return
		}
		epoch = n
	}
	id := r.PathValue("id")
	a, rc, err := h.svc.Model(r.Context(), id, epoch)
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	defer rc.Close()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.FormatInt(a.SizeBytes, 10))
	w.Header().Set("Content-Disposition", `attachment; filename="`+id+"-"+a.Name+`"`)
	_, _ = io.Copy(w, rc)
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
