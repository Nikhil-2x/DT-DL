package dataset

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"time"
	"unicode"

	"dtdl/backend/internal/apperr"
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

// SetUsageChecker wires in the check that blocks deleting in-use datasets.
func (s *Service) SetUsageChecker(u UsageChecker) { s.usage = u }

type UploadInput struct {
	Name     string // optional; defaults to the filename
	Filename string
	Body     io.Reader
}

// Upload streams the body into object storage and records the dataset. The
// body is never fully loaded into memory.
func (s *Service) Upload(ctx context.Context, in UploadInput) (Dataset, error) {
	safeName, display, err := SanitizeFilename(in.Filename)
	if err != nil {
		return Dataset{}, err
	}
	name, err := validateName(in.Name, display)
	if err != nil {
		return Dataset{}, err
	}

	id := ids.New()
	now := s.now().UTC()
	d := Dataset{
		ID: id, Name: name, OriginalFilename: display,
		Bucket: s.bucket, Key: id + "/" + safeName,
		Status: StatusUploading, CreatedAt: now, UpdatedAt: now,
	}
	if err := s.repo.Create(ctx, d); err != nil {
		return Dataset{}, apperr.Internal(err)
	}

	body := &limitReader{r: in.Body, remaining: s.maxBytes}
	size, err := s.store.Put(ctx, d.Bucket, d.Key, body, -1, "application/octet-stream")
	switch {
	case body.exceeded:
		s.abort(d)
		return Dataset{}, apperr.TooLarge("UPLOAD_TOO_LARGE", "upload exceeds the maximum allowed size")
	case err != nil:
		s.abort(d)
		s.log.Error("dataset upload failed", "event", "dataset_upload_failed", "dataset_id", id, "error", err)
		return Dataset{}, apperr.Internal(err)
	case size == 0:
		s.abort(d)
		return Dataset{}, apperr.Invalid("EMPTY_FILE", "uploaded file is empty")
	}

	d.SizeBytes, d.Status, d.UpdatedAt = size, StatusReady, s.now().UTC()
	if err := s.repo.Update(ctx, d); err != nil {
		s.abort(d)
		return Dataset{}, apperr.Internal(err)
	}
	s.log.Info("dataset uploaded", "event", "dataset_uploaded", "dataset_id", id, "size_bytes", size)
	return d, nil
}

// Download returns the stored dataset object for streaming to the client.
func (s *Service) Download(ctx context.Context, id string) (Dataset, io.ReadCloser, error) {
	d, err := s.Get(ctx, id)
	if err != nil {
		return Dataset{}, nil, err
	}

	if d.Status != StatusReady {
		return Dataset{}, nil, apperr.Conflict(
			"DATASET_NOT_READY",
			"dataset is not ready for download",
		)
	}

	obj, err := s.store.Get(ctx, d.Bucket, d.Key)
	if errors.Is(err, storage.ErrNotFound) {
		return Dataset{}, nil, apperr.NotFound(
			"DATASET_OBJECT_NOT_FOUND",
			"dataset object not found in storage",
		)
	}
	if err != nil {
		return Dataset{}, nil, apperr.Internal(err)
	}

	return d, obj, nil
}

// abort marks the dataset FAILED and removes any partial object. It uses a
// fresh context because the request context is often what got cancelled.
func (s *Service) abort(d Dataset) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = s.store.Delete(ctx, d.Bucket, d.Key)
	d.Status, d.UpdatedAt = StatusFailed, s.now().UTC()
	if err := s.repo.Update(ctx, d); err != nil {
		s.log.Error("mark dataset failed", "dataset_id", d.ID, "error", err)
	}
}

func (s *Service) Get(ctx context.Context, id string) (Dataset, error) {
	d, err := s.repo.Get(ctx, id)
	if errors.Is(err, ErrNotFound) {
		return Dataset{}, apperr.NotFound("DATASET_NOT_FOUND", "dataset not found")
	}
	if err != nil {
		return Dataset{}, apperr.Internal(err)
	}
	return d, nil
}

func (s *Service) List(ctx context.Context) ([]Dataset, error) {
	ds, err := s.repo.List(ctx)
	if err != nil {
		return nil, apperr.Internal(err)
	}
	return ds, nil
}

// Delete removes the dataset's objects and its metadata record.
func (s *Service) Delete(ctx context.Context, id string) error {
	d, err := s.Get(ctx, id)
	if err != nil {
		return err
	}
	if d.Status == StatusUploading {
		return apperr.Conflict("DATASET_UPLOADING", "dataset upload is still in progress")
	}
	if s.usage != nil {
		inUse, err := s.usage.DatasetInUse(ctx, id)
		if err != nil {
			return apperr.Internal(err)
		}
		if inUse {
			return apperr.Conflict("DATASET_IN_USE", "dataset is used by an unfinished training job")
		}
	}
	objs, err := s.store.List(ctx, d.Bucket, d.Prefix())
	if err != nil {
		return apperr.Internal(err)
	}
	for _, o := range objs {
		if err := s.store.Delete(ctx, d.Bucket, o.Key); err != nil {
			return apperr.Internal(err)
		}
	}
	if err := s.repo.Delete(ctx, id); err != nil && !errors.Is(err, ErrNotFound) {
		return apperr.Internal(err)
	}
	s.log.Info("dataset deleted", "event", "dataset_deleted", "dataset_id", id)
	return nil
}

// RecoverInterrupted marks uploads left UPLOADING by a previous process as
// FAILED. Call once at startup, before serving requests.
func (s *Service) RecoverInterrupted(ctx context.Context) error {
	ds, err := s.repo.List(ctx)
	if err != nil {
		return err
	}
	for _, d := range ds {
		if d.Status == StatusUploading {
			d.Status, d.UpdatedAt = StatusFailed, s.now().UTC()
			if err := s.repo.Update(ctx, d); err != nil {
				return err
			}
			s.log.Warn("interrupted upload marked failed", "event", "dataset_upload_interrupted", "dataset_id", d.ID)
		}
	}
	return nil
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

// limitReader fails once more than `remaining` bytes have been read, and
// records that it did so, whatever error the storage layer wraps it in.
type limitReader struct {
	r         io.Reader
	remaining int64
	exceeded  bool
}

func (l *limitReader) Read(p []byte) (int, error) {
	if l.exceeded {
		return 0, errTooLarge
	}
	n, err := l.r.Read(p)
	l.remaining -= int64(n)
	if l.remaining < 0 {
		l.exceeded = true
		return 0, errTooLarge
	}
	return n, err
}

var errTooLarge = errors.New("upload exceeds maximum size")
