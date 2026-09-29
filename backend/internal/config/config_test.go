package config

import (
	"strings"
	"testing"
)

func env(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func TestLoadDefaults(t *testing.T) {
	c, err := Load(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if c.ServerPort != "8080" || c.Runner != "mock" || c.DatasetBucket != "datasets" || c.CheckpointBucket != "checkpoints" {
		t.Fatalf("unexpected defaults: %+v", c)
	}
}

func TestLoadOverridesAndErrors(t *testing.T) {
	c, err := Load(env(map[string]string{"SERVER_PORT": "9000", "RUNNER": "DOCKER", "S3_ENDPOINT": "http://s3:8333"}))
	if err != nil || c.ServerPort != "9000" || c.Runner != "docker" || c.S3Endpoint != "http://s3:8333" {
		t.Fatalf("got %+v, %v", c, err)
	}
	_, err = Load(env(map[string]string{"RUNNER": "k8s", "MAX_UPLOAD_BYTES": "-1", "SERVER_PORT": "x"}))
	if err == nil {
		t.Fatal("expected error")
	}
	for _, want := range []string{"RUNNER", "MAX_UPLOAD_BYTES", "SERVER_PORT"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q should mention %s", err, want)
		}
	}
}
