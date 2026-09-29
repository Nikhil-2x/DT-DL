// Package handlers is the HTTP layer: it decodes requests, calls the
// dataset/job services and renders JSON. It contains no business logic.
package handlers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/http/middleware"
)

type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func statusFor(k apperr.Kind) int {
	switch k {
	case apperr.KindInvalid:
		return http.StatusBadRequest
	case apperr.KindNotFound:
		return http.StatusNotFound
	case apperr.KindConflict:
		return http.StatusConflict
	case apperr.KindTooLarge:
		return http.StatusRequestEntityTooLarge
	default:
		return http.StatusInternalServerError
	}
}

// writeError renders any error as the standard JSON error body. Internal
// causes are logged (with the request ID) but never sent to the client.
func writeError(w http.ResponseWriter, r *http.Request, log *slog.Logger, err error) {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		err = apperr.TooLarge("REQUEST_TOO_LARGE", "request body is too large")
	}
	ae := apperr.From(err)
	if ae.Kind == apperr.KindInternal {
		log.Error("request failed", "request_id", middleware.RequestID(r.Context()), "error", ae.Err)
	}
	writeJSON(w, statusFor(ae.Kind), errorBody{Error: errorDetail{Code: ae.Code, Message: ae.Message}})
}

const maxJSONBody = 1 << 20

// decodeJSON reads a strict JSON body: size-limited, unknown fields rejected.
func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return err
		}
		return apperr.Invalid("INVALID_JSON", "request body must be valid JSON with known fields only")
	}
	if dec.More() {
		return apperr.Invalid("INVALID_JSON", "request body must contain a single JSON object")
	}
	return nil
}
