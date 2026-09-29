// Package dataset holds the dataset model and the service that uploads,
// lists and deletes datasets. Handlers call this service; it in turn talks to
// the ObjectStorage abstraction and never to S3 directly.
package dataset

import (
	"context"
	"errors"
	"time"
)

type Status string

const (
	StatusUploading Status = "UPLOADING"
	StatusReady     Status = "READY"
	StatusFailed    Status = "FAILED"
)

type Dataset struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	OriginalFilename string    `json:"original_filename"`
	Bucket           string    `json:"bucket"`
	Key              string    `json:"key"`
	SizeBytes        int64     `json:"size_bytes"`
	Status           Status    `json:"status"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// Prefix is the object-key prefix that holds everything belonging to this
// dataset. It is what `train.py --dataset-prefix` expects.
func (d Dataset) Prefix() string { return d.ID + "/" }

var ErrNotFound = errors.New("dataset not found")

type Repository interface {
	Create(ctx context.Context, d Dataset) error
	Get(ctx context.Context, id string) (Dataset, error) // ErrNotFound if missing
	List(ctx context.Context) ([]Dataset, error)         // newest first
	Update(ctx context.Context, d Dataset) error         // ErrNotFound if missing
	Delete(ctx context.Context, id string) error         // ErrNotFound if missing
}

// UsageChecker reports whether a dataset is referenced by unfinished jobs.
// It is satisfied by the job service and injected at startup, so this
// package does not depend on the job package.
type UsageChecker interface {
	DatasetInUse(ctx context.Context, datasetID string) (bool, error)
}
