package dataset_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/repository"
	"dtdl/backend/internal/storage"
)

type fixture struct {
	svc   *dataset.Service
	store *storage.Memory
	repo  *repository.DatasetRepo
}

func newFixture(t *testing.T, maxBytes int64) fixture {
	t.Helper()
	db, err := repository.Open("sqlite::memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	store := storage.NewMemory()
	repo := repository.NewDatasetRepo(db)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return fixture{dataset.NewService(repo, store, "datasets", maxBytes, log), store, repo}
}

func kind(t *testing.T, err error) (apperr.Kind, string) {
	t.Helper()
	var ae *apperr.Error
	if !errors.As(err, &ae) {
		t.Fatalf("want *apperr.Error, got %T %v", err, err)
	}
	return ae.Kind, ae.Code
}

func TestUploadCreatesMetadataAndObject(t *testing.T) {
	f := newFixture(t, 1<<20)
	d, err := f.svc.Upload(context.Background(), dataset.UploadInput{Name: "My Data", Filename: "train data.csv", Body: strings.NewReader("a,b\n1,2\n")})
	if err != nil {
		t.Fatal(err)
	}
	if d.Status != dataset.StatusReady || d.SizeBytes != 8 || d.Name != "My Data" || d.OriginalFilename != "train data.csv" {
		t.Fatalf("unexpected dataset: %+v", d)
	}
	if d.Bucket != "datasets" || d.Key != d.ID+"/train_data.csv" || d.Prefix() != d.ID+"/" {
		t.Fatalf("bad storage location: %+v", d)
	}
	if objs, _ := f.store.List(context.Background(), "datasets", d.Prefix()); len(objs) != 1 || objs[0].Key != d.Key {
		t.Fatalf("object not stored: %+v", objs)
	}
	got, err := f.svc.Get(context.Background(), d.ID)
	if err != nil || got.Status != dataset.StatusReady {
		t.Fatalf("persisted record: %+v %v", got, err)
	}
}

func TestUploadDefaultsNameToFilename(t *testing.T) {
	f := newFixture(t, 1<<20)
	d, err := f.svc.Upload(context.Background(), dataset.UploadInput{Filename: "x.bin", Body: strings.NewReader("1")})
	if err != nil || d.Name != "x.bin" {
		t.Fatalf("%+v %v", d, err)
	}
}

func TestUploadValidation(t *testing.T) {
	f := newFixture(t, 10)
	cases := []struct {
		name     string
		in       dataset.UploadInput
		wantKind apperr.Kind
		wantCode string
	}{
		{"no filename", dataset.UploadInput{Filename: "", Body: strings.NewReader("x")}, apperr.KindInvalid, "INVALID_FILENAME"},
		{"dotdot", dataset.UploadInput{Filename: "..", Body: strings.NewReader("x")}, apperr.KindInvalid, "INVALID_FILENAME"},
		{"control chars", dataset.UploadInput{Filename: "a\x00b", Body: strings.NewReader("x")}, apperr.KindInvalid, "INVALID_FILENAME"},
		{"long name", dataset.UploadInput{Filename: "a.csv", Name: strings.Repeat("n", 101), Body: strings.NewReader("x")}, apperr.KindInvalid, "INVALID_NAME"},
		{"empty file", dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("")}, apperr.KindInvalid, "EMPTY_FILE"},
		{"too large", dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader(strings.Repeat("x", 11))}, apperr.KindTooLarge, "UPLOAD_TOO_LARGE"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := f.svc.Upload(context.Background(), c.in)
			k, code := kind(t, err)
			if k != c.wantKind || code != c.wantCode {
				t.Fatalf("got kind=%v code=%s (%v)", k, code, err)
			}
		})
	}
	// Nothing rejected may linger as an object.
	if objs, _ := f.store.List(context.Background(), "datasets", ""); len(objs) != 0 {
		t.Fatalf("leftover objects: %+v", objs)
	}
}

func TestUploadStorageFailureMarksFailed(t *testing.T) {
	f := newFixture(t, 1<<20)
	f.store.PutErr = errors.New("seaweed down")
	_, err := f.svc.Upload(context.Background(), dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})
	if k, _ := kind(t, err); k != apperr.KindInternal {
		t.Fatalf("want internal, got %v", err)
	}
	ds, _ := f.svc.List(context.Background())
	if len(ds) != 1 || ds[0].Status != dataset.StatusFailed {
		t.Fatalf("want one FAILED record, got %+v", ds)
	}
}

func TestDelete(t *testing.T) {
	f := newFixture(t, 1<<20)
	ctx := context.Background()
	d, _ := f.svc.Upload(ctx, dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})
	if err := f.svc.Delete(ctx, d.ID); err != nil {
		t.Fatal(err)
	}
	if objs, _ := f.store.List(ctx, "datasets", d.Prefix()); len(objs) != 0 {
		t.Fatalf("object not deleted: %+v", objs)
	}
	_, err := f.svc.Get(ctx, d.ID)
	if k, code := kind(t, err); k != apperr.KindNotFound || code != "DATASET_NOT_FOUND" {
		t.Fatalf("got %v", err)
	}
	if k, _ := kind(t, f.svc.Delete(ctx, d.ID)); k != apperr.KindNotFound {
		t.Fatal("second delete should be not found")
	}
}

type inUse bool

func (u inUse) DatasetInUse(context.Context, string) (bool, error) { return bool(u), nil }

func TestDeleteBlockedWhenInUse(t *testing.T) {
	f := newFixture(t, 1<<20)
	f.svc.SetUsageChecker(inUse(true))
	ctx := context.Background()
	d, _ := f.svc.Upload(ctx, dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})
	if k, code := kind(t, f.svc.Delete(ctx, d.ID)); k != apperr.KindConflict || code != "DATASET_IN_USE" {
		t.Fatalf("got kind=%v code=%s", k, code)
	}
}

func TestRecoverInterrupted(t *testing.T) {
	f := newFixture(t, 1<<20)
	ctx := context.Background()
	stuck := dataset.Dataset{ID: "stuck", Name: "s", OriginalFilename: "s", Bucket: "datasets", Key: "stuck/s", Status: dataset.StatusUploading}
	if err := f.repo.Create(ctx, stuck); err != nil {
		t.Fatal(err)
	}
	if err := f.svc.RecoverInterrupted(ctx); err != nil {
		t.Fatal(err)
	}
	if got, _ := f.svc.Get(ctx, "stuck"); got.Status != dataset.StatusFailed {
		t.Fatalf("got %s", got.Status)
	}
}

func TestSanitizeFilename(t *testing.T) {
	cases := map[string]string{
		"data.csv":                 "data.csv",
		"../../etc/passwd":         "passwd",
		`C:\Users\me\data set.zip`: "data_set.zip",
		".hidden":                  "hidden",
		"we ird/naïve;name.tar.gz": "na_ve_name.tar.gz",
		"a/b/../c.bin":             "c.bin",
	}
	for in, want := range cases {
		got, _, err := dataset.SanitizeFilename(in)
		if err != nil || got != want {
			t.Errorf("SanitizeFilename(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"", ".", "..", "\x01x"} {
		if _, _, err := dataset.SanitizeFilename(bad); err == nil {
			t.Errorf("SanitizeFilename(%q) should fail", bad)
		}
	}
}
