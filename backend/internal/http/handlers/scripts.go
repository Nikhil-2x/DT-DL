package handlers

import (
	"errors"
	"io"
	"log/slog"
	"net/http"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/script"
)

type Scripts struct {
	svc       *script.Service
	log       *slog.Logger
	maxUpload int64
}

func NewScripts(svc *script.Service, maxUpload int64, log *slog.Logger) *Scripts {
	return &Scripts{svc: svc, log: log, maxUpload: maxUpload}
}

// Upload handles multipart/form-data with an optional "name" field and a
// "file" part holding the .py project script ("name" before "file").
func (h *Scripts) Upload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.maxUpload+(1<<20))
	mr, err := r.MultipartReader()
	if err != nil {
		writeError(w, r, h.log, apperr.Invalid("INVALID_MULTIPART", "expected a multipart/form-data body"))
		return
	}
	var name string
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			writeError(w, r, h.log, mapBodyErr(err))
			return
		}
		switch part.FormName() {
		case "name":
			b, err := io.ReadAll(io.LimitReader(part, 1024))
			if err != nil {
				writeError(w, r, h.log, mapBodyErr(err))
				return
			}
			name = string(b)
		case "file":
			sc, err := h.svc.Upload(r.Context(), script.UploadInput{Name: name, Filename: part.FileName(), Body: part})
			if err != nil {
				writeError(w, r, h.log, err)
				return
			}
			writeJSON(w, http.StatusCreated, sc)
			return
		}
	}
	writeError(w, r, h.log, apperr.Invalid("MISSING_FILE", `multipart field "file" is required`))
}

func (h *Scripts) List(w http.ResponseWriter, r *http.Request) {
	out, err := h.svc.List(r.Context())
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"scripts": out})
}

func (h *Scripts) Get(w http.ResponseWriter, r *http.Request) {
	sc, err := h.svc.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, sc)
}

// Content returns the stored Python source as plain text.
func (h *Scripts) Content(w http.ResponseWriter, r *http.Request) {
	rc, err := h.svc.Content(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	defer rc.Close()
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = io.Copy(w, rc)
}

func (h *Scripts) Delete(w http.ResponseWriter, r *http.Request) {
	if err := h.svc.Delete(r.Context(), r.PathValue("id")); err != nil {
		writeError(w, r, h.log, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// Template serves a runnable example that follows the plug-in contract.
func (h *Scripts) Template(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = io.WriteString(w, script.Template)
}
