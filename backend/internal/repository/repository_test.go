package repository

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/job"
)

func TestJobRepoRoundTrip(t *testing.T) {
	db, err := Open("sqlite:" + filepath.Join(t.TempDir(), "sub", "t.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	r := NewJobRepo(db)
	ctx := context.Background()

	now := time.Now().UTC().Truncate(time.Millisecond)
	j := job.Job{ID: "j1", Name: "n", Image: "i", Entrypoint: "train.py", DataMode: "synthetic", Workers: 2, Epochs: 3,
		Status: job.StatusPending, CheckpointLocation: "s3://c/j1/", CreatedAt: now, UpdatedAt: now}
	if err := r.Create(ctx, j); err != nil {
		t.Fatal(err)
	}
	j.Status, j.StartedAt = job.StatusRunning, &now
	if err := r.Update(ctx, j); err != nil {
		t.Fatal(err)
	}
	got, err := r.Get(ctx, "j1")
	if err != nil || got.Status != job.StatusRunning || got.StartedAt == nil || !got.StartedAt.Equal(now) || got.FinishedAt != nil || got.Workers != 2 {
		t.Fatalf("%+v %v", got, err)
	}
	if _, err := r.Get(ctx, "missing"); !errors.Is(err, job.ErrNotFound) {
		t.Fatalf("got %v", err)
	}
	if err := r.Update(ctx, job.Job{ID: "missing"}); !errors.Is(err, job.ErrNotFound) {
		t.Fatalf("got %v", err)
	}
	if js, _ := r.ListByStatus(ctx, job.StatusRunning, job.StatusQueued); len(js) != 1 {
		t.Fatalf("%+v", js)
	}
	if js, _ := r.ListByStatus(ctx, job.StatusFailed); len(js) != 0 {
		t.Fatalf("%+v", js)
	}
}

func TestMigrationsAreIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.db")
	for i := 0; i < 2; i++ {
		db, err := Open(path)
		if err != nil {
			t.Fatalf("open #%d: %v", i, err)
		}
		db.Close()
	}
}

func TestDatasetRepoNotFound(t *testing.T) {
	db, _ := Open("sqlite::memory:")
	defer db.Close()
	r := NewDatasetRepo(db)
	if _, err := r.Get(context.Background(), "x"); !errors.Is(err, dataset.ErrNotFound) {
		t.Fatalf("got %v", err)
	}
	if err := r.Delete(context.Background(), "x"); !errors.Is(err, dataset.ErrNotFound) {
		t.Fatalf("got %v", err)
	}
}

func TestPostgresURLRejected(t *testing.T) {
	if _, err := Open("postgres://u:p@h/db"); err == nil {
		t.Fatal("expected error")
	}
}
