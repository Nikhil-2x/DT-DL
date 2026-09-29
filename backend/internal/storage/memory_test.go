package storage

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

// The same behavioural contract is exercised against real SeaweedFS in
// s3_integration_test.go.
func testContract(t *testing.T, s ObjectStorage, bucket string) {
	ctx := context.Background()
	if err := s.EnsureBucket(ctx, bucket); err != nil {
		t.Fatal(err)
	}
	if err := s.EnsureBucket(ctx, bucket); err != nil {
		t.Fatalf("EnsureBucket must be idempotent: %v", err)
	}
	n, err := s.Put(ctx, bucket, "p/a.txt", strings.NewReader("hello"), -1, "text/plain")
	if err != nil || n != 5 {
		t.Fatalf("put: n=%d err=%v", n, err)
	}
	if _, err := s.Put(ctx, bucket, "p/b.txt", strings.NewReader("x"), -1, "text/plain"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Put(ctx, bucket, "q/c.txt", strings.NewReader("y"), -1, "text/plain"); err != nil {
		t.Fatal(err)
	}

	rc, err := s.Get(ctx, bucket, "p/a.txt")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(rc)
	rc.Close()
	if string(data) != "hello" {
		t.Fatalf("got %q", data)
	}
	if _, err := s.Get(ctx, bucket, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("want ErrNotFound, got %v", err)
	}

	objs, err := s.List(ctx, bucket, "p/")
	if err != nil || len(objs) != 2 || objs[0].Key != "p/a.txt" {
		t.Fatalf("list: %+v %v", objs, err)
	}
	if err := s.Delete(ctx, bucket, "p/a.txt"); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx, bucket, "p/a.txt"); err != nil {
		t.Fatalf("delete must be idempotent: %v", err)
	}
	if objs, _ := s.List(ctx, bucket, "p/"); len(objs) != 1 {
		t.Fatalf("after delete: %+v", objs)
	}
}

func TestMemoryContract(t *testing.T) { testContract(t, NewMemory(), "b") }
