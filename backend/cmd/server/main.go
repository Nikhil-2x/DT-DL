package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"dtdl/backend/internal/config"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/http/handlers"
	"dtdl/backend/internal/job"
	"dtdl/backend/internal/repository"
	"dtdl/backend/internal/runner"
	"dtdl/backend/internal/storage"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load(os.Getenv)
	if err != nil {
		return err
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: cfg.LogLevel}))

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	db, err := repository.Open(cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer db.Close()

	store, err := storage.NewS3(storage.S3Config{
		Endpoint: cfg.S3Endpoint, AccessKey: cfg.S3AccessKey, SecretKey: cfg.S3SecretKey, Region: cfg.S3Region,
	})
	if err != nil {
		return err
	}
	for _, b := range []string{cfg.DatasetBucket, cfg.CheckpointBucket} {
		bctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		err := store.EnsureBucket(bctx, b)
		cancel()
		if err != nil {
			return err
		}
	}

	var jr runner.JobRunner
	switch cfg.Runner {
	case "docker":
		jr = runner.NewDocker(runner.DockerConfig{
			S3Endpoint: cfg.RunnerS3Endpoint, S3AccessKey: cfg.S3AccessKey, S3SecretKey: cfg.S3SecretKey,
		})
	default:
		jr = runner.NewMock(cfg.MockEpochDuration)
	}
	log.Info("runner selected", "runner", cfg.Runner)

	datasets := dataset.NewService(repository.NewDatasetRepo(db), store, cfg.DatasetBucket, cfg.MaxUploadBytes, log)
	jobs := job.NewService(repository.NewJobRepo(db), datasets, jr, job.Config{
		Image: cfg.TrainingImage, Entrypoint: "train.py",
		CheckpointBucket: cfg.CheckpointBucket, MaxWorkers: cfg.MaxWorkers,
	}, log)
	datasets.SetUsageChecker(jobs)

	if err := datasets.RecoverInterrupted(ctx); err != nil {
		return err
	}

	go jobs.Run(ctx, cfg.ReconcileInterval)

	srv := &http.Server{
		Addr: net.JoinHostPort("", cfg.ServerPort),
		Handler: handlers.NewRouter(handlers.Deps{
			Datasets: datasets, Jobs: jobs, MaxUploadBytes: cfg.MaxUploadBytes, Log: log,
		}),
		// No Read/WriteTimeout: they would cut off large uploads and log
		// streams. Slow-header attacks are covered by ReadHeaderTimeout.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	errc := make(chan error, 1)
	go func() {
		log.Info("server listening", "addr", srv.Addr)
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
		log.Info("shutting down")
		sctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return srv.Shutdown(sctx)
	}
	return nil
}
