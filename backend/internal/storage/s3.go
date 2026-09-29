package storage

import (
	"context"
	"fmt"
	"io"
	"net/url"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// multipartPartSize bounds memory per upload when the object size is unknown:
// the client buffers one part at a time, never the whole object.
const multipartPartSize = 16 << 20

// S3 implements ObjectStorage against any S3-compatible server (SeaweedFS).
// minio-go is used purely as an S3 client library.
type S3 struct {
	client *minio.Client
}

type S3Config struct {
	Endpoint  string // e.g. http://127.0.0.1:8333
	AccessKey string
	SecretKey string
	Region    string
}

func NewS3(cfg S3Config) (*S3, error) {
	u, err := url.Parse(cfg.Endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, fmt.Errorf("invalid S3 endpoint %q: want http(s)://host:port", cfg.Endpoint)
	}
	c, err := minio.New(u.Host, &minio.Options{
		Creds:        credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure:       u.Scheme == "https",
		Region:       cfg.Region,
		BucketLookup: minio.BucketLookupPath, // SeaweedFS is addressed path-style
	})
	if err != nil {
		return nil, fmt.Errorf("create S3 client: %w", err)
	}
	return &S3{client: c}, nil
}

func (s *S3) EnsureBucket(ctx context.Context, bucket string) error {
	exists, err := s.client.BucketExists(ctx, bucket)
	if err != nil {
		return fmt.Errorf("check bucket %q: %w", bucket, err)
	}
	if exists {
		return nil
	}
	if err := s.client.MakeBucket(ctx, bucket, minio.MakeBucketOptions{}); err != nil {
		code := minio.ToErrorResponse(err).Code
		if code == "BucketAlreadyOwnedByYou" || code == "BucketAlreadyExists" {
			return nil
		}
		return fmt.Errorf("create bucket %q: %w", bucket, err)
	}
	return nil
}

func (s *S3) Put(ctx context.Context, bucket, key string, r io.Reader, size int64, contentType string) (int64, error) {
	info, err := s.client.PutObject(ctx, bucket, key, r, size, minio.PutObjectOptions{
		ContentType: contentType,
		PartSize:    multipartPartSize,
	})
	if err != nil {
		return 0, fmt.Errorf("put %s/%s: %w", bucket, key, err)
	}
	return info.Size, nil
}

func (s *S3) Get(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
	obj, err := s.client.GetObject(ctx, bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("get %s/%s: %w", bucket, key, err)
	}
	// GetObject is lazy; Stat surfaces "no such key" now.
	if _, err := obj.Stat(); err != nil {
		obj.Close()
		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("stat %s/%s: %w", bucket, key, err)
	}
	return obj, nil
}

func (s *S3) Delete(ctx context.Context, bucket, key string) error {
	if err := s.client.RemoveObject(ctx, bucket, key, minio.RemoveObjectOptions{}); err != nil {
		return fmt.Errorf("delete %s/%s: %w", bucket, key, err)
	}
	return nil
}

func (s *S3) List(ctx context.Context, bucket, prefix string) ([]ObjectInfo, error) {
	var out []ObjectInfo
	for obj := range s.client.ListObjects(ctx, bucket, minio.ListObjectsOptions{Prefix: prefix, Recursive: true}) {
		if obj.Err != nil {
			return nil, fmt.Errorf("list %s/%s: %w", bucket, prefix, obj.Err)
		}
		out = append(out, ObjectInfo{Key: obj.Key, Size: obj.Size, LastModified: obj.LastModified})
	}
	return out, nil
}
