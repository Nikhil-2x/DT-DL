package job_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"dtdl/backend/internal/apperr"
	"dtdl/backend/internal/dataset"
	"dtdl/backend/internal/job"
	"dtdl/backend/internal/repository"
	"dtdl/backend/internal/runner"
	"dtdl/backend/internal/storage"
)

func TestTransitionTable(t *testing.T) {
	ok := [][2]job.Status{
		{job.StatusPending, job.StatusQueued},
		{job.StatusQueued, job.StatusRunning}, {job.StatusQueued, job.StatusFailed}, {job.StatusQueued, job.StatusStopping},
		{job.StatusRunning, job.StatusCompleted}, {job.StatusRunning, job.StatusFailed}, {job.StatusRunning, job.StatusStopping},
		{job.StatusStopping, job.StatusStopped}, {job.StatusStopping, job.StatusCompleted}, {job.StatusStopping, job.StatusFailed},
	}
	for _, p := range ok {
		if !job.CanTransition(p[0], p[1]) {
			t.Errorf("%s -> %s should be allowed", p[0], p[1])
		}
	}
	bad := [][2]job.Status{
		{job.StatusPending, job.StatusRunning}, {job.StatusPending, job.StatusCompleted},
		{job.StatusCompleted, job.StatusRunning}, {job.StatusFailed, job.StatusQueued},
		{job.StatusStopped, job.StatusRunning}, {job.StatusRunning, job.StatusStopped},
		{job.StatusRunning, job.StatusQueued},
	}
	for _, p := range bad {
		if job.CanTransition(p[0], p[1]) {
			t.Errorf("%s -> %s should be rejected", p[0], p[1])
		}
	}
	for _, s := range []job.Status{job.StatusCompleted, job.StatusFailed, job.StatusStopped} {
		if !s.Terminal() {
			t.Errorf("%s should be terminal", s)
		}
	}
}

// fakeRunner is a scriptable JobRunner.
type fakeRunner struct {
	mu        sync.Mutex
	started   []runner.TrainingSpec
	stopped   []string
	startErr  error
	status    map[string]runner.Status
	missing   map[string]bool
	stopCalls int
}

func newFakeRunner() *fakeRunner {
	return &fakeRunner{status: map[string]runner.Status{}, missing: map[string]bool{}}
}

func (f *fakeRunner) Start(_ context.Context, s runner.TrainingSpec) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.startErr != nil {
		return f.startErr
	}
	f.started = append(f.started, s)
	f.status[s.JobID] = runner.Status{State: runner.StateRunning}
	return nil
}
func (f *fakeRunner) Stop(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopCalls++
	f.stopped = append(f.stopped, id)
	return nil
}
func (f *fakeRunner) Status(_ context.Context, id string) (runner.Status, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.missing[id] {
		return runner.Status{}, runner.ErrNotFound
	}
	st, ok := f.status[id]
	if !ok {
		return runner.Status{}, runner.ErrNotFound
	}
	return st, nil
}
func (f *fakeRunner) Logs(context.Context, string, int) (io.ReadCloser, error) {
	return io.NopCloser(strings.NewReader("log line\n")), nil
}
func (f *fakeRunner) set(id string, st runner.Status) { f.mu.Lock(); f.status[id] = st; f.mu.Unlock() }

type fixture struct {
	svc      *job.Service
	datasets *dataset.Service
	runner   *fakeRunner
}

func newFixture(t *testing.T) fixture {
	t.Helper()
	db, err := repository.Open("sqlite::memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	ds := dataset.NewService(repository.NewDatasetRepo(db), storage.NewMemory(), "datasets", 1<<20, log)
	fr := newFakeRunner()
	svc := job.NewService(repository.NewJobRepo(db), ds, fr, job.Config{
		Image: "ml-runner:dev", Entrypoint: "train.py", CheckpointBucket: "checkpoints", MaxWorkers: 8,
	}, log)
	ds.SetUsageChecker(svc)
	return fixture{svc, ds, fr}
}

func code(t *testing.T, err error) string {
	t.Helper()
	var ae *apperr.Error
	if !errors.As(err, &ae) {
		t.Fatalf("want *apperr.Error, got %T %v", err, err)
	}
	return ae.Code
}

func TestCreateJob(t *testing.T) {
	f := newFixture(t)
	j, err := f.svc.Create(context.Background(), job.CreateInput{Name: "cifar-demo", Epochs: 3, Workers: 4})
	if err != nil {
		t.Fatal(err)
	}
	if j.Status != job.StatusPending || j.DataMode != "synthetic" || j.Image != "ml-runner:dev" || j.Entrypoint != "train.py" {
		t.Fatalf("unexpected job: %+v", j)
	}
	if j.CheckpointLocation != "s3://checkpoints/"+j.ID+"/" || j.Workers != 4 {
		t.Fatalf("unexpected job: %+v", j)
	}
	got, err := f.svc.Get(context.Background(), j.ID)
	if err != nil || got.ID != j.ID {
		t.Fatalf("not persisted: %v", err)
	}
}

func TestCreateDefaultsWorkersToOne(t *testing.T) {
	f := newFixture(t)
	j, err := f.svc.Create(context.Background(), job.CreateInput{Name: "x", Epochs: 1})
	if err != nil || j.Workers != 1 {
		t.Fatalf("%+v %v", j, err)
	}
}

func TestCreateValidation(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	up, _ := f.datasets.Upload(ctx, dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})

	cases := []struct {
		name string
		in   job.CreateInput
		want string
	}{
		{"empty name", job.CreateInput{Epochs: 1}, "INVALID_NAME"},
		{"shell name", job.CreateInput{Name: "x; rm -rf /", Epochs: 1}, "INVALID_NAME"},
		{"zero epochs", job.CreateInput{Name: "x"}, "INVALID_EPOCHS"},
		{"huge epochs", job.CreateInput{Name: "x", Epochs: 5000}, "INVALID_EPOCHS"},
		{"too many workers", job.CreateInput{Name: "x", Epochs: 1, Workers: 99}, "INVALID_WORKERS"},
		{"negative workers", job.CreateInput{Name: "x", Epochs: 1, Workers: -1}, "INVALID_WORKERS"},
		{"bad gpus", job.CreateInput{Name: "x", Epochs: 1, GPUs: 100}, "INVALID_RESOURCES"},
		{"bad mode", job.CreateInput{Name: "x", Epochs: 1, DataMode: "evil"}, "INVALID_DATA_MODE"},
		{"sharded w/o dataset", job.CreateInput{Name: "x", Epochs: 1, DataMode: "sharded"}, "INVALID_DATA_MODE"},
		{"dataset with synthetic", job.CreateInput{Name: "x", Epochs: 1, DataMode: "synthetic", DatasetID: up.ID}, "INVALID_DATA_MODE"},
		{"missing dataset", job.CreateInput{Name: "x", Epochs: 1, DatasetID: "nope"}, "DATASET_NOT_FOUND"},
		{"sharded multi-worker", job.CreateInput{Name: "x", Epochs: 1, DatasetID: up.ID, Workers: 2}, "INVALID_WORKERS"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := f.svc.Create(ctx, c.in); err == nil || code(t, err) != c.want {
				t.Fatalf("want %s, got %v", c.want, err)
			}
		})
	}
	if js, _ := f.svc.List(ctx); len(js) != 0 {
		t.Fatalf("invalid requests created jobs: %+v", js)
	}
}

func TestStartBuildsStructuredSpec(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	up, _ := f.datasets.Upload(ctx, dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})
	j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 2, DatasetID: up.ID, MemoryMB: 512})

	started, err := f.svc.Start(ctx, j.ID)
	if err != nil {
		t.Fatal(err)
	}
	if started.Status != job.StatusRunning || started.StartedAt == nil {
		t.Fatalf("%+v", started)
	}
	s := f.runner.started[0]
	if s.RunID != j.ID || s.DataMode != "sharded" || s.DatasetBucket != "datasets" || s.DatasetPrefix != up.ID+"/" ||
		s.CheckpointBucket != "checkpoints" || s.Resources.MemoryMB != 512 || s.Entrypoint != "train.py" {
		t.Fatalf("bad spec: %+v", s)
	}
}

func TestStartTwiceConflicts(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 1})
	if _, err := f.svc.Start(ctx, j.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.svc.Start(ctx, j.ID); err == nil || code(t, err) != "INVALID_JOB_STATE" {
		t.Fatalf("got %v", err)
	}
	if len(f.runner.started) != 1 {
		t.Fatal("runner started twice")
	}
}

func TestStartRunnerFailureMarksFailed(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	f.runner.startErr = errors.New("docker daemon not running")
	j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 1})
	got, err := f.svc.Start(ctx, j.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != job.StatusFailed || !strings.Contains(got.Error, "docker daemon not running") || got.FinishedAt == nil {
		t.Fatalf("%+v", got)
	}
}

func TestStopFlow(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 1})

	if _, err := f.svc.Stop(ctx, j.ID); err == nil || code(t, err) != "INVALID_JOB_STATE" {
		t.Fatalf("stopping PENDING should conflict, got %v", err)
	}
	f.svc.Start(ctx, j.ID)
	stopping, err := f.svc.Stop(ctx, j.ID)
	if err != nil || stopping.Status != job.StatusStopping {
		t.Fatalf("%+v %v", stopping, err)
	}
	// Runner still reports RUNNING: reconcile re-issues the stop.
	if err := f.svc.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if f.runner.stopCalls != 2 {
		t.Fatalf("expected stop retry, calls=%d", f.runner.stopCalls)
	}
	// Runner confirms termination.
	f.runner.set(j.ID, runner.Status{State: runner.StateFailed, Message: "stopped"})
	f.svc.Reconcile(ctx)
	got, _ := f.svc.Get(ctx, j.ID)
	if got.Status != job.StatusStopped || got.FinishedAt == nil || got.Error != "" {
		t.Fatalf("%+v", got)
	}
	if _, err := f.svc.Stop(ctx, j.ID); err == nil {
		t.Fatal("stopping a STOPPED job should conflict")
	}
}

func TestReconcileOutcomes(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	mk := func() job.Job {
		j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 1})
		f.svc.Start(ctx, j.ID)
		return j
	}
	ok, failed, lost, running := mk(), mk(), mk(), mk()
	f.runner.set(ok.ID, runner.Status{State: runner.StateSucceeded})
	f.runner.set(failed.ID, runner.Status{State: runner.StateFailed, Message: "training exited with code 1"})
	f.runner.missing[lost.ID] = true

	if err := f.svc.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	check := func(id string, want job.Status) job.Job {
		t.Helper()
		j, _ := f.svc.Get(ctx, id)
		if j.Status != want {
			t.Fatalf("job %s: want %s got %s (%s)", id, want, j.Status, j.Error)
		}
		return j
	}
	done := check(ok.ID, job.StatusCompleted)
	if done.OutputLocation != done.CheckpointLocation || done.FinishedAt == nil {
		t.Fatalf("%+v", done)
	}
	if j := check(failed.ID, job.StatusFailed); j.Error != "training exited with code 1" {
		t.Fatalf("error=%q", j.Error)
	}
	check(lost.ID, job.StatusFailed)
	check(running.ID, job.StatusRunning)
}

func TestDatasetInUseBlocksDelete(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()
	up, _ := f.datasets.Upload(ctx, dataset.UploadInput{Filename: "a.csv", Body: strings.NewReader("x")})
	j, _ := f.svc.Create(ctx, job.CreateInput{Name: "x", Epochs: 1, DatasetID: up.ID})

	if err := f.datasets.Delete(ctx, up.ID); err == nil || code(t, err) != "DATASET_IN_USE" {
		t.Fatalf("got %v", err)
	}
	f.svc.Start(ctx, j.ID)
	f.runner.set(j.ID, runner.Status{State: runner.StateSucceeded})
	f.svc.Reconcile(ctx)
	if err := f.datasets.Delete(ctx, up.ID); err != nil {
		t.Fatalf("finished job should not block delete: %v", err)
	}
}

func TestRunLoopStopsOnCancel(t *testing.T) {
	f := newFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { f.svc.Run(ctx, 5*time.Millisecond); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not return after cancel")
	}
}
