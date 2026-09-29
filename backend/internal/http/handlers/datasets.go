package handlers

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
)

type Datasets struct {
	svc       *dataset.Service
	log       *slog.Logger
	maxUpload int64
}

func (h *Datasets) Download(w http.ResponseWriter, r *http.Request) {
	d, obj, err := h.svc.Download(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	defer obj.Close()

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set(
		"Content-Disposition",
		`attachment; filename="`+d.OriginalFilename+`"`,
	)

	if d.SizeBytes >= 0 {
		w.Header().Set("Content-Length", strconv.FormatInt(d.SizeBytes, 10))
	}
	if _, err := io.Copy(w, obj); err != nil {
		h.log.Error(
			"dataset download failed",
			"event", "dataset_download_failed",
			"dataset_id", d.ID,
			"error", err,
		)
	}
}
func NewDatasets(svc *dataset.Service, maxUpload int64, log *slog.Logger) *Datasets {
	return &Datasets{svc: svc, log: log, maxUpload: maxUpload}
}

// Upload handles multipart/form-data with an optional "name" text field and a
// "file" part. Parts are streamed straight through to object storage; the
// form is never parsed into memory or spooled to disk. "name" must be sent
// before "file".
func (h *Datasets) Upload(w http.ResponseWriter, r *http.Request) {
	// Slack over the file limit covers multipart framing and the name field.
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
			d, err := h.svc.Upload(r.Context(), dataset.UploadInput{Name: name, Filename: part.FileName(), Body: part})
			if err != nil {
				writeError(w, r, h.log, err)
				return
			}
			writeJSON(w, http.StatusCreated, d)
			return
		}
	}
	writeError(w, r, h.log, apperr.Invalid("MISSING_FILE", `multipart field "file" is required`))
}

func mapBodyErr(err error) error {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		return apperr.TooLarge("UPLOAD_TOO_LARGE", "upload exceeds the maximum allowed size")
	}
	return apperr.Invalid("INVALID_MULTIPART", "malformed multipart body")
}

func (h *Datasets) List(w http.ResponseWriter, r *http.Request) {
	ds, err := h.svc.List(r.Context())
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"datasets": ds})
}

func (h *Datasets) Get(w http.ResponseWriter, r *http.Request) {
	d, err := h.svc.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		writeError(w, r, h.log, err)
		return
	}
	writeJSON(w, http.StatusOK, d)
}

func (h *Datasets) Delete(w http.ResponseWriter, r *http.Request) {
	if err := h.svc.Delete(r.Context(), r.PathValue("id")); err != nil {
		writeError(w, r, h.log, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
