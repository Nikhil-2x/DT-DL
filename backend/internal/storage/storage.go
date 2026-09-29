// Package storage defines the object-storage abstraction used by the rest of
// the backend. Nothing outside this package should know the backend is
// SeaweedFS or that it is spoken to over the S3 API.
package storage

import (
	"context"
	"errors"
	"io"
	"time"
)

var ErrNotFound = errors.New("object not found")

type ObjectInfo struct {
	Key          string
	Size         int64
	LastModified time.Time
}

type ObjectStorage interface {
	// EnsureBucket creates the bucket if it does not exist.
	EnsureBucket(ctx context.Context, bucket string) error
	// Put streams r into the object. size may be -1 when unknown; the
	// implementation must not buffer the whole object. It returns the number
	// of bytes stored.
	Put(ctx context.Context, bucket, key string, r io.Reader, size int64, contentType string) (int64, error)
	// Get returns ErrNotFound if the object does not exist.
	Get(ctx context.Context, bucket, key string) (io.ReadCloser, error)
	// Delete is idempotent: deleting a missing object is not an error.
	Delete(ctx context.Context, bucket, key string) error
	// List returns all objects whose key starts with prefix.
	List(ctx context.Context, bucket, prefix string) ([]ObjectInfo, error)
}
