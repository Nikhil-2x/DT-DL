package storage

import (
	"bytes"
	"context"
	"io"
	"sort"
	"strings"
	"sync"
	"time"
)

// Memory is an in-memory ObjectStorage for tests and demos. It buffers
// objects, so it is NOT suitable for real uploads.
type Memory struct {
	mu      sync.Mutex
	objects map[string]map[string][]byte
	// PutErr, if set, is returned by Put after the reader has been consumed.
	PutErr error
}

func NewMemory() *Memory { return &Memory{objects: map[string]map[string][]byte{}} }

func (m *Memory) EnsureBucket(_ context.Context, bucket string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.objects[bucket] == nil {
		m.objects[bucket] = map[string][]byte{}
	}
	return nil
}

func (m *Memory) Put(_ context.Context, bucket, key string, r io.Reader, _ int64, _ string) (int64, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return 0, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.PutErr != nil {
		return 0, m.PutErr
	}
	if m.objects[bucket] == nil {
		m.objects[bucket] = map[string][]byte{}
	}
	m.objects[bucket][key] = data
	return int64(len(data)), nil
}

func (m *Memory) Get(_ context.Context, bucket, key string) (io.ReadCloser, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	data, ok := m.objects[bucket][key]
	if !ok {
		return nil, ErrNotFound
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

func (m *Memory) Delete(_ context.Context, bucket, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.objects[bucket], key)
	return nil
}

func (m *Memory) List(_ context.Context, bucket, prefix string) ([]ObjectInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []ObjectInfo
	for k, v := range m.objects[bucket] {
		if strings.HasPrefix(k, prefix) {
			out = append(out, ObjectInfo{Key: k, Size: int64(len(v)), LastModified: time.Now()})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}
