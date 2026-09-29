// Package config loads backend settings from environment variables.
package config

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	ServerPort  string
	DatabaseURL string
	LogLevel    slog.Level

	S3Endpoint       string
	S3AccessKey      string
	S3SecretKey      string
	S3Region         string
	DatasetBucket    string
	CheckpointBucket string
	MaxUploadBytes   int64

	// Runner is "mock" or "docker".
	Runner            string
	TrainingImage     string
	RunnerS3Endpoint  string // S3 endpoint as seen from inside training containers
	MaxWorkers        int
	MockEpochDuration time.Duration
	ReconcileInterval time.Duration
}

// Load reads configuration using getenv (os.Getenv in production).
func Load(getenv func(string) string) (Config, error) {
	var errs []string
	str := func(key, def string) string {
		if v := strings.TrimSpace(getenv(key)); v != "" {
			return v
		}
		return def
	}
	integer := func(key string, def int64) int64 {
		v := strings.TrimSpace(getenv(key))
		if v == "" {
			return def
		}
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n <= 0 {
			errs = append(errs, fmt.Sprintf("%s must be a positive integer", key))
			return def
		}
		return n
	}
	duration := func(key string, def time.Duration) time.Duration {
		v := strings.TrimSpace(getenv(key))
		if v == "" {
			return def
		}
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			errs = append(errs, fmt.Sprintf("%s must be a positive duration like 2s", key))
			return def
		}
		return d
	}

	c := Config{
		ServerPort:  str("SERVER_PORT", "8080"),
		DatabaseURL: str("DATABASE_URL", "sqlite:./data/backend.db"),

		S3Endpoint: str("S3_ENDPOINT", "http://127.0.0.1:8333"),
		// A default SeaweedFS started with `server -s3` has auth disabled and
		// accepts any key pair; "any" mirrors what the Python code uses.
		S3AccessKey: str("S3_ACCESS_KEY", "dtdl-access"),
S3SecretKey: str("S3_SECRET_KEY", "dtdl-secret"),
		S3Region:         str("S3_REGION", "us-east-1"),
		DatasetBucket:    str("DATASET_BUCKET", "datasets"),
		CheckpointBucket: str("CHECKPOINT_BUCKET", "checkpoints"),
		MaxUploadBytes:   integer("MAX_UPLOAD_BYTES", 10<<30),

		Runner:            strings.ToLower(str("RUNNER", "mock")),
		TrainingImage:     str("TRAINING_IMAGE", "ml-runner:dev"),
		MaxWorkers:        int(integer("MAX_WORKERS", 8)),
		MockEpochDuration: duration("MOCK_EPOCH_DURATION", 2*time.Second),
		ReconcileInterval: duration("RECONCILE_INTERVAL", 2*time.Second),
	}
	c.RunnerS3Endpoint = str("RUNNER_S3_ENDPOINT", "http://host.docker.internal:8333")

	if c.Runner != "mock" && c.Runner != "docker" {
		errs = append(errs, "RUNNER must be \"mock\" or \"docker\"")
	}
	if _, err := strconv.Atoi(c.ServerPort); err != nil {
		errs = append(errs, "SERVER_PORT must be a number")
	}
	switch strings.ToLower(str("LOG_LEVEL", "info")) {
	case "debug":
		c.LogLevel = slog.LevelDebug
	case "info":
		c.LogLevel = slog.LevelInfo
	case "warn":
		c.LogLevel = slog.LevelWarn
	case "error":
		c.LogLevel = slog.LevelError
	default:
		errs = append(errs, "LOG_LEVEL must be debug, info, warn or error")
	}
	if len(errs) > 0 {
		return Config{}, fmt.Errorf("invalid configuration: %s", strings.Join(errs, "; "))
	}
	return c, nil
}
