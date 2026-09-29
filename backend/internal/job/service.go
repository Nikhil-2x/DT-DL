package job

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/ids"
	"dtdl/backend/internal/runner"
)

// DatasetLookup is the slice of the dataset service the job service needs.
type DatasetLookup interface {
	Get(ctx context.Context, id string) (dataset.Dataset, error)
}

type Config struct {
	Image            string // training image; not client-selectable
	Entrypoint       string // training script; not client-selectable
	CheckpointBucket string
	MaxWorkers       int
}

type Service struct {
	repo     Repository
	datasets DatasetLookup
	runner   runner.JobRunner
	cfg      Config
	log      *slog.Logger
	now      func() time.Time

	// mu serialises state changes (API calls and the reconcile loop) so two
	// callers cannot both act on the same stale status. Fine for one backend
	// instance; a multi-replica control plane needs DB-level compare-and-swap.
	mu sync.Mutex
}

func NewService(repo Repository, datasets DatasetLookup, r runner.JobRunner, cfg Config, log *slog.Logger) *Service {
	return &Service{repo: repo, datasets: datasets, runner: r, cfg: cfg, log: log, now: time.Now}
}

type CreateInput struct {
	Name      string
	DatasetID string
	Epochs    int
	Workers   int    // default 1
	DataMode  string // default: sharded if DatasetID set, else synthetic
	GPUs      int
	CPUCores  int
	MemoryMB  int
}

var nameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 ._-]{0,99}$`)

func (s *Service) Create(ctx context.Context, in CreateInput) (Job, error) {
	if !nameRe.MatchString(in.Name) {
		return Job{}, apperr.Invalid("INVALID_NAME", "name must be 1-100 characters: letters, digits, space, '.', '_' or '-'")
	}
	if in.Epochs < 1 || in.Epochs > 1000 {
		return Job{}, apperr.Invalid("INVALID_EPOCHS", "epochs must be between 1 and 1000")
	}
	if in.Workers == 0 {
		in.Workers = 1
	}
	if in.Workers < 1 || in.Workers > s.cfg.MaxWorkers {
		return Job{}, apperr.Invalid("INVALID_WORKERS", fmt.Sprintf("workers must be between 1 and %d", s.cfg.MaxWorkers))
	}
	if in.GPUs < 0 || in.GPUs > 8 || in.CPUCores < 0 || in.CPUCores > 256 || in.MemoryMB < 0 || in.MemoryMB > 4<<20 {
		return Job{}, apperr.Invalid("INVALID_RESOURCES", "gpus, cpu_cores or memory_mb out of range")
	}
	if in.DataMode == "" {
		in.DataMode = "synthetic"
		if in.DatasetID != "" {
			in.DataMode = "sharded"
		}
	}
	switch in.DataMode {
	case "synthetic", "cifar10":
		if in.DatasetID != "" {
			return Job{}, apperr.Invalid("INVALID_DATA_MODE", "dataset_id can only be used with data_mode \"sharded\"")
		}
	case "sharded":
		if in.DatasetID == "" {
			return Job{}, apperr.Invalid("INVALID_DATA_MODE", "data_mode \"sharded\" requires dataset_id")
		}
		ds, err := s.datasets.Get(ctx, in.DatasetID)
		if err != nil {
			return Job{}, err
		}
		if ds.Status != dataset.StatusReady {
			return Job{}, apperr.Conflict("DATASET_NOT_READY", "dataset is not ready")
		}
		// train.py's sharded mode gives each rank a disjoint slice of the
		// objects under the prefix; a rank with no files would hang DDP's
		// collectives. Uploads are single objects for now, so 1 worker only.
		if in.Workers > 1 {
			return Job{}, apperr.Invalid("INVALID_WORKERS", "sharded datasets currently hold a single object; use workers=1")
		}
	default:
		return Job{}, apperr.Invalid("INVALID_DATA_MODE", "data_mode must be synthetic, cifar10 or sharded")
	}

	now := s.now().UTC()
	id := ids.New()
	j := Job{
		ID: id, Name: in.Name, DatasetID: in.DatasetID,
		Image: s.cfg.Image, Entrypoint: s.cfg.Entrypoint, DataMode: in.DataMode,
		Workers: in.Workers, Epochs: in.Epochs, GPUs: in.GPUs, CPUCores: in.CPUCores, MemoryMB: in.MemoryMB,
		Status:             StatusPending,
		CheckpointLocation: fmt.Sprintf("s3://%s/%s/", s.cfg.CheckpointBucket, id),
		CreatedAt:          now, UpdatedAt: now,
	}
	if err := s.repo.Create(ctx, j); err != nil {
		return Job{}, apperr.Internal(err)
	}
	s.event(j, "job_created")
	return j, nil
}

func (s *Service) Get(ctx context.Context, id string) (Job, error) {
	j, err := s.repo.Get(ctx, id)
	if errors.Is(err, ErrNotFound) {
		return Job{}, apperr.NotFound("JOB_NOT_FOUND", "job not found")
	}
	if err != nil {
		return Job{}, apperr.Internal(err)
	}
	return j, nil
}

func (s *Service) List(ctx context.Context) ([]Job, error) {
	js, err := s.repo.List(ctx)
	if err != nil {
		return nil, apperr.Internal(err)
	}
	return js, nil
}

// Start moves PENDING -> QUEUED, hands the job to the runner, then
// QUEUED -> RUNNING (or FAILED if the runner rejects it).
func (s *Service) Start(ctx context.Context, id string) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	j, err := s.Get(ctx, id)
	if err != nil {
		return Job{}, err
	}
	if j.Status != StatusPending {
		return Job{}, apperr.Conflict("INVALID_JOB_STATE", fmt.Sprintf("cannot start a job in state %s", j.Status))
	}
	spec, err := s.buildSpec(ctx, j)
	if err != nil {
		return Job{}, err
	}
	if err := s.move(ctx, &j, StatusQueued, ""); err != nil {
		return Job{}, err
	}
	s.event(j, "job_queued")

	if err := s.runner.Start(ctx, spec); err != nil {
		s.log.Error("runner start failed", "job_id", j.ID, "event", "job_start_failed", "error", err)
		if merr := s.move(ctx, &j, StatusFailed, "failed to start: "+err.Error()); merr != nil {
			return Job{}, merr
		}
		s.event(j, "job_failed")
		return j, nil // the job record reflects the failure
	}
	if err := s.move(ctx, &j, StatusRunning, ""); err != nil {
		return Job{}, err
	}
	s.event(j, "job_started")
	return j, nil
}

// Stop moves a QUEUED/RUNNING job to STOPPING and asks the runner to stop
// it. Reconcile moves it to STOPPED once the runner confirms.
func (s *Service) Stop(ctx context.Context, id string) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	j, err := s.Get(ctx, id)
	if err != nil {
		return Job{}, err
	}
	if j.Status != StatusRunning && j.Status != StatusQueued {
		return Job{}, apperr.Conflict("INVALID_JOB_STATE", fmt.Sprintf("cannot stop a job in state %s", j.Status))
	}
	if err := s.move(ctx, &j, StatusStopping, ""); err != nil {
		return Job{}, err
	}
	s.event(j, "job_stopping")
	if err := s.runner.Stop(ctx, j.ID); err != nil {
		// Stay STOPPING; Reconcile retries the stop.
		s.log.Error("runner stop failed", "job_id", j.ID, "event", "job_stop_failed", "error", err)
		return Job{}, apperr.Internal(err)
	}
	return j, nil
}

// Logs returns the job's log stream. Jobs that never started have none.
func (s *Service) Logs(ctx context.Context, id string, tail int) (io.ReadCloser, error) {
	j, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	if j.Status == StatusPending {
		return io.NopCloser(strings.NewReader("")), nil
	}
	rc, err := s.runner.Logs(ctx, j.ID, tail)
	if errors.Is(err, runner.ErrNotFound) {
		return io.NopCloser(strings.NewReader("")), nil
	}
	if err != nil {
		return nil, apperr.Internal(err)
	}
	return rc, nil
}

// DatasetInUse implements dataset.UsageChecker.
func (s *Service) DatasetInUse(ctx context.Context, datasetID string) (bool, error) {
	n, err := s.repo.CountUnfinishedByDataset(ctx, datasetID)
	return n > 0, err
}

// Run reconciles job state against the runner until ctx is cancelled.
func (s *Service) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := s.Reconcile(ctx); err != nil && ctx.Err() == nil {
				s.log.Error("reconcile failed", "event", "reconcile_failed", "error", err)
			}
		}
	}
}

// Reconcile does one pass: it asks the runner about every unfinished job and
// applies the resulting transition. Polling keeps the runner interface
// simple; a Kubernetes runner could later push events instead.
func (s *Service) Reconcile(ctx context.Context) error {
	jobs, err := s.repo.ListByStatus(ctx, StatusQueued, StatusRunning, StatusStopping)
	if err != nil {
		return err
	}
	for _, j := range jobs {
		if err := s.reconcileOne(ctx, j.ID); err != nil {
			s.log.Error("reconcile job", "job_id", j.ID, "event", "reconcile_job_failed", "error", err)
		}
	}
	return nil
}

func (s *Service) reconcileOne(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	j, err := s.repo.Get(ctx, id) // re-read under the lock
	if err != nil {
		return err
	}
	if j.Status.Terminal() || j.Status == StatusPending {
		return nil
	}
	st, err := s.runner.Status(ctx, j.ID)
	missing := errors.Is(err, runner.ErrNotFound)
	if err != nil && !missing {
		return err
	}

	switch {
	case j.Status == StatusStopping:
		if !missing && st.State == runner.StateRunning {
			return s.runner.Stop(ctx, j.ID) // earlier stop did not land; retry
		}
		return s.finish(ctx, &j, StatusStopped, "job_stopped", "")
	case missing:
		return s.finish(ctx, &j, StatusFailed, "job_failed", "runner lost track of the job (backend restarted before it launched?)")
	case st.State == runner.StateSucceeded:
		j.OutputLocation = j.CheckpointLocation
		return s.finish(ctx, &j, StatusCompleted, "job_completed", "")
	case st.State == runner.StateFailed:
		return s.finish(ctx, &j, StatusFailed, "job_failed", st.Message)
	case st.State == runner.StateRunning && j.Status == StatusQueued:
		if err := s.move(ctx, &j, StatusRunning, ""); err != nil {
			return err
		}
		s.event(j, "job_running")
	}
	return nil
}

func (s *Service) finish(ctx context.Context, j *Job, to Status, event, errMsg string) error {
	if err := s.move(ctx, j, to, errMsg); err != nil {
		return err
	}
	s.event(*j, event)
	return nil
}

// move applies a validated state transition and persists it.
func (s *Service) move(ctx context.Context, j *Job, to Status, errMsg string) error {
	if !CanTransition(j.Status, to) {
		return apperr.Conflict("INVALID_JOB_STATE", fmt.Sprintf("illegal transition %s -> %s", j.Status, to))
	}
	now := s.now().UTC()
	j.Status, j.UpdatedAt = to, now
	if to == StatusRunning {
		j.StartedAt = &now
	}
	if to.Terminal() {
		j.FinishedAt = &now
	}
	if errMsg != "" {
		j.Error = errMsg
	}
	if err := s.repo.Update(ctx, *j); err != nil {
		return apperr.Internal(err)
	}
	return nil
}

func (s *Service) buildSpec(ctx context.Context, j Job) (runner.TrainingSpec, error) {
	spec := runner.TrainingSpec{
		JobID: j.ID, RunID: j.ID, Image: j.Image, Entrypoint: j.Entrypoint,
		Workers: j.Workers, Epochs: j.Epochs, DataMode: j.DataMode,
		CheckpointBucket: s.cfg.CheckpointBucket,
		Resources:        runner.Resources{CPUCores: j.CPUCores, MemoryMB: j.MemoryMB, GPUs: j.GPUs},
	}
	if j.DataMode == "sharded" {
		ds, err := s.datasets.Get(ctx, j.DatasetID)
		if err != nil {
			return spec, err
		}
		if ds.Status != dataset.StatusReady {
			return spec, apperr.Conflict("DATASET_NOT_READY", "dataset is not ready")
		}
		spec.DatasetBucket, spec.DatasetPrefix = ds.Bucket, ds.Prefix()
	}
	if err := spec.Validate(); err != nil {
		return spec, apperr.Invalid("INVALID_SPEC", err.Error())
	}
	return spec, nil
}

// event logs a lifecycle event with the job id, e.g.
// job_id=abc123 event=job_started.
func (s *Service) event(j Job, event string) {
	s.log.Info(event, "job_id", j.ID, "event", event, "status", string(j.Status))
}
