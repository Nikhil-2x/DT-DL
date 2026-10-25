package script

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/ids"
	"dtdl/backend/internal/storage"
)

const maxNameLen = 100

type Service struct {
	repo     Repository
	store    storage.ObjectStorage
	bucket   string
	maxBytes int64
	usage    UsageChecker
	log      *slog.Logger
	now      func() time.Time
}

func NewService(repo Repository, store storage.ObjectStorage, bucket string, maxBytes int64, log *slog.Logger) *Service {
	return &Service{repo: repo, store: store, bucket: bucket, maxBytes: maxBytes, log: log, now: time.Now}
}

// SetUsageChecker wires in the check that blocks deleting in-use scripts.
func (s *Service) SetUsageChecker(u UsageChecker) { s.usage = u }

type UploadInput struct {
	Name     string // optional; defaults to the filename
	Filename string
	Body     io.Reader
}

// requiredFuncs are the names the contract requires. The check below is only
// a cheap early filter for obvious mistakes (wrong file); the training
// container performs the authoritative validation by importing the file.
var requiredFuncs = []string{"build_model", "get_dataset"}

var definitionRes = func() map[string]*regexp.Regexp {
	m := map[string]*regexp.Regexp{}
	for _, n := range requiredFuncs {
		// top-level `def name(`, `name = ...`, or `from x import ... name`
		m[n] = regexp.MustCompile(`(?m)^(?:def\s+` + n + `\s*\(|` + n + `\s*=[^=]|from\s+\S+\s+import\s+.*\b` + n + `\b)`)
	}
	return m
}()

// Validate checks that src looks like a contract-following Python file.
func Validate(src []byte) error {
	if len(src) == 0 {
		return apperr.Invalid("EMPTY_FILE", "uploaded file is empty")
	}
	if !utf8.Valid(src) || bytes.IndexByte(src, 0) >= 0 {
		return apperr.Invalid("INVALID_SCRIPT", "script must be a UTF-8 text file")
	}
	var missing []string
	for _, n := range requiredFuncs {
		if !definitionRes[n].Match(src) {
			missing = append(missing, n+"()")
		}
	}
	if len(missing) > 0 {
		return apperr.Invalid("INVALID_SCRIPT",
			"script must define "+strings.Join(missing, " and ")+" at top level (see GET /api/v1/scripts/template)")
	}
	return nil
}

// Upload validates and stores a script. Scripts are small, so the body is read
// into memory (bounded by maxBytes) rather than streamed.
func (s *Service) Upload(ctx context.Context, in UploadInput) (Script, error) {
	_, display, err := dataset.SanitizeFilename(in.Filename)
	if err != nil {
		return Script{}, err
	}
	if !strings.HasSuffix(strings.ToLower(display), ".py") {
		return Script{}, apperr.Invalid("INVALID_SCRIPT_FILE", "script must be a .py file")
	}
	name, err := validateName(in.Name, display)
	if err != nil {
		return Script{}, err
	}

	src, err := io.ReadAll(io.LimitReader(in.Body, s.maxBytes+1))
	if err != nil {
		return Script{}, apperr.Internal(err)
	}
	if int64(len(src)) > s.maxBytes {
		return Script{}, apperr.TooLarge("SCRIPT_TOO_LARGE", fmt.Sprintf("script exceeds the maximum size of %d bytes", s.maxBytes))
	}
	if err := Validate(src); err != nil {
		return Script{}, err
	}

	id := ids.New()
	sc := Script{
		ID: id, Name: name, OriginalFilename: display,
		Bucket: s.bucket, Key: id + "/" + ObjectName, SizeBytes: int64(len(src)),
		CreatedAt: s.now().UTC(),
	}
	if _, err := s.store.Put(ctx, sc.Bucket, sc.Key, bytes.NewReader(src), int64(len(src)), "text/x-python"); err != nil {
		s.log.Error("script upload failed", "event", "script_upload_failed", "script_id", id, "error", err)
		return Script{}, apperr.Internal(err)
	}
	if err := s.repo.Create(ctx, sc); err != nil {
		s.deleteObject(sc)
		return Script{}, apperr.Internal(err)
	}
	s.log.Info("script uploaded", "event", "script_uploaded", "script_id", id, "size_bytes", sc.SizeBytes)
	return sc, nil
}

func (s *Service) Get(ctx context.Context, id string) (Script, error) {
	sc, err := s.repo.Get(ctx, id)
	if errors.Is(err, ErrNotFound) {
		return Script{}, apperr.NotFound("SCRIPT_NOT_FOUND", "script not found")
	}
	if err != nil {
		return Script{}, apperr.Internal(err)
	}
	return sc, nil
}

func (s *Service) List(ctx context.Context) ([]Script, error) {
	out, err := s.repo.List(ctx)
	if err != nil {
		return nil, apperr.Internal(err)
	}
	return out, nil
}

// Content returns the stored source of a script.
func (s *Service) Content(ctx context.Context, id string) (io.ReadCloser, error) {
	sc, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	rc, err := s.store.Get(ctx, sc.Bucket, sc.Key)
	if errors.Is(err, storage.ErrNotFound) {
		return nil, apperr.NotFound("SCRIPT_NOT_FOUND", "script file is missing from storage")
	}
	if err != nil {
		return nil, apperr.Internal(err)
	}
	return rc, nil
}

// Delete removes the script object and its record. Scripts used by unfinished
// jobs cannot be deleted.
func (s *Service) Delete(ctx context.Context, id string) error {
	sc, err := s.Get(ctx, id)
	if err != nil {
		return err
	}
	if s.usage != nil {
		inUse, err := s.usage.ScriptInUse(ctx, id)
		if err != nil {
			return apperr.Internal(err)
		}
		if inUse {
			return apperr.Conflict("SCRIPT_IN_USE", "script is used by an unfinished training job")
		}
	}
	if err := s.store.Delete(ctx, sc.Bucket, sc.Key); err != nil {
		return apperr.Internal(err)
	}
	if err := s.repo.Delete(ctx, id); err != nil && !errors.Is(err, ErrNotFound) {
		return apperr.Internal(err)
	}
	s.log.Info("script deleted", "event", "script_deleted", "script_id", id)
	return nil
}

func (s *Service) deleteObject(sc Script) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := s.store.Delete(ctx, sc.Bucket, sc.Key); err != nil {
		s.log.Error("cleanup script object", "script_id", sc.ID, "error", err)
	}
}

func validateName(name, fallback string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		name = fallback
	}
	if len([]rune(name)) > maxNameLen {
		return "", apperr.Invalid("INVALID_NAME", "name must be at most 100 characters")
	}
	for _, r := range name {
		if unicode.IsControl(r) {
			return "", apperr.Invalid("INVALID_NAME", "name contains control characters")
		}
	}
	return name, nil
}
