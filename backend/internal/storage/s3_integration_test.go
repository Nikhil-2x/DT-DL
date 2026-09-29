package storage

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// Runs only when S3_TEST_ENDPOINT is set, e.g.
//
//	S3_TEST_ENDPOINT=http://127.0.0.1:8333 go test ./internal/storage -run S3
func TestS3Contract(t *testing.T) {
	endpoint := os.Getenv("S3_TEST_ENDPOINT")
	if endpoint == "" {
		t.Skip("set S3_TEST_ENDPOINT to run against a real S3-compatible server")
	}
	s, err := NewS3(S3Config{Endpoint: endpoint, AccessKey: envOr("S3_ACCESS_KEY", "any"), SecretKey: envOr("S3_SECRET_KEY", "any"), Region: "us-east-1"})
	if err != nil {
		t.Fatal(err)
	}
	testContract(t, s, fmt.Sprintf("it-%d", time.Now().UnixNano()))
}

func TestNewS3RejectsBadEndpoint(t *testing.T) {
	for _, e := range []string{"", "127.0.0.1:8333", "ftp://x"} {
		if _, err := NewS3(S3Config{Endpoint: e}); err == nil {
			t.Errorf("endpoint %q should be rejected", e)
		}
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
