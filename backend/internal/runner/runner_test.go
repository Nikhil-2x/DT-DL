package runner

import (
	"context"
	"errors"
	"io"
	"slices"
	"strings"
	"testing"
	"time"
)

func validSpec() TrainingSpec {
	return TrainingSpec{
		JobID: "abc123", RunID: "abc123", Image: "ml-runner:dev", Entrypoint: "train.py",
		Workers: 4, Epochs: 3, DataMode: "synthetic", CheckpointBucket: "checkpoints",
	}
}

func TestSpecValidate(t *testing.T) {
	if err := validSpec().Validate(); err != nil {
		t.Fatal(err)
	}
	mutations := map[string]func(*TrainingSpec){
		"shell in run id":   func(s *TrainingSpec) { s.RunID = "x; rm -rf /" },
		"flag-like run id":  func(s *TrainingSpec) { s.RunID = "--storage" },
		"flag-like image":   func(s *TrainingSpec) { s.Image = "--privileged" },
		"space in image":    func(s *TrainingSpec) { s.Image = "img --net=host" },
		"unknown script":    func(s *TrainingSpec) { s.Entrypoint = "evil.py" },
		"bad data mode":     func(s *TrainingSpec) { s.DataMode = "x" },
		"zero workers":      func(s *TrainingSpec) { s.Workers = 0 },
		"zero epochs":       func(s *TrainingSpec) { s.Epochs = 0 },
		"negative cpu":      func(s *TrainingSpec) { s.Resources.CPUCores = -1 },
		"sharded no prefix": func(s *TrainingSpec) { s.DataMode = "sharded" },
	}
	for name, mut := range mutations {
		s := validSpec()
		mut(&s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
}

func TestDockerRunArgsMatchesReadmeCommand(t *testing.T) {
	d := NewDocker(DockerConfig{S3Endpoint: "http://host.docker.internal:8333", S3AccessKey: "ak", S3SecretKey: "sk"})
	s := validSpec()
	s.Resources = Resources{CPUCores: 2, MemoryMB: 2048}
	args := d.RunArgs(s)

	tail := args[slices.Index(args, "ml-runner:dev"):]
	want := []string{"ml-runner:dev", "torchrun", "--nproc_per_node=4", "train.py",
		"--epochs", "3", "--run-id", "abc123", "--storage", "s3",
		"--s3-endpoint", "http://host.docker.internal:8333", "--s3-bucket", "checkpoints",
		"--s3-access-key", "ak", "--s3-secret-key", "sk", "--data", "synthetic"}
	if !slices.Equal(tail, want) {
		t.Fatalf("got  %v\nwant %v", tail, want)
	}
	if !slices.Contains(args, "--cpus") || !slices.Contains(args, "2048m") {
		t.Fatalf("resource limits missing: %v", args)
	}
}

func TestDockerRunArgsSharded(t *testing.T) {
	d := NewDocker(DockerConfig{})
	s := validSpec()
	s.DataMode, s.DatasetBucket, s.DatasetPrefix = "sharded", "datasets", "ds1/"
	args := d.RunArgs(s)
	i := slices.Index(args, "--dataset-prefix")
	if i < 0 || args[i+1] != "ds1/" || !slices.Contains(args, "--dataset-bucket") {
		t.Fatalf("dataset args missing: %v", args)
	}
}

type call struct {
	name string
	args []string
}

func fakeExec(out string, err error, calls *[]call) ExecFunc {
	return func(_ context.Context, name string, args ...string) ([]byte, error) {
		*calls = append(*calls, call{name, args})
		return []byte(out), err
	}
}

func TestDockerStartUsesArgvNotShell(t *testing.T) {
	var calls []call
	d := NewDockerWithExec(DockerConfig{}, fakeExec("", nil, &calls))
	if err := d.Start(context.Background(), validSpec()); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].name != "docker" || calls[0].args[0] != "run" {
		t.Fatalf("calls: %+v", calls)
	}
	// An invalid spec must never reach exec.
	calls = nil
	bad := validSpec()
	bad.RunID = "$(reboot)"
	if err := d.Start(context.Background(), bad); err == nil || len(calls) != 0 {
		t.Fatalf("invalid spec reached exec: err=%v calls=%v", err, calls)
	}
	gpu := validSpec()
	gpu.Resources.GPUs = 1
	if err := d.Start(context.Background(), gpu); err == nil {
		t.Fatal("gpus should be rejected by docker runner")
	}
}

func TestDockerStatus(t *testing.T) {
	cases := []struct {
		out   string
		err   error
		state State
		gone  bool
	}{
		{"running 0\n", nil, StateRunning, false},
		{"exited 0\n", nil, StateSucceeded, false},
		{"exited 1\n", nil, StateFailed, false},
		{"exited 137\n", nil, StateFailed, false},
		{"", errors.New("Error: No such object: dtdl-job-x"), "", true},
	}
	for _, c := range cases {
		var calls []call
		d := NewDockerWithExec(DockerConfig{}, fakeExec(c.out, c.err, &calls))
		st, err := d.Status(context.Background(), "x")
		if c.gone {
			if !errors.Is(err, ErrNotFound) {
				t.Errorf("want ErrNotFound, got %v", err)
			}
			continue
		}
		if err != nil || st.State != c.state {
			t.Errorf("%q: got %+v, %v; want %s", c.out, st, err, c.state)
		}
	}
}

func TestDockerStopMissingContainerIsOK(t *testing.T) {
	var calls []call
	d := NewDockerWithExec(DockerConfig{}, fakeExec("", errors.New("No such container"), &calls))
	if err := d.Stop(context.Background(), "x"); err != nil {
		t.Fatal(err)
	}
}

func TestMockLifecycle(t *testing.T) {
	now := time.Unix(1000, 0)
	m := NewMock(time.Second)
	m.Now = func() time.Time { return now }
	ctx := context.Background()
	s := validSpec()
	s.Workers, s.Epochs = 2, 3

	if _, err := m.Status(ctx, "abc123"); !errors.Is(err, ErrNotFound) {
		t.Fatal("want not found before start")
	}
	if err := m.Start(ctx, s); err != nil {
		t.Fatal(err)
	}
	if err := m.Start(ctx, s); err == nil {
		t.Fatal("double start should fail")
	}
	if st, _ := m.Status(ctx, "abc123"); st.State != StateRunning {
		t.Fatalf("got %+v", st)
	}

	now = now.Add(2 * time.Second)
	rc, _ := m.Logs(ctx, "abc123", 0)
	logs, _ := io.ReadAll(rc)
	if got := strings.Count(string(logs), "epoch "); got != 4 { // 2 epochs x 2 ranks
		t.Fatalf("want 4 epoch lines, got %d:\n%s", got, logs)
	}

	now = now.Add(2 * time.Second)
	if st, _ := m.Status(ctx, "abc123"); st.State != StateSucceeded {
		t.Fatalf("got %+v", st)
	}
}

func TestMockStop(t *testing.T) {
	m := NewMock(time.Hour)
	ctx := context.Background()
	if err := m.Start(ctx, validSpec()); err != nil {
		t.Fatal(err)
	}
	if err := m.Stop(ctx, "abc123"); err != nil {
		t.Fatal(err)
	}
	if st, _ := m.Status(ctx, "abc123"); st.State != StateFailed {
		t.Fatalf("got %+v", st)
	}
	if err := m.Stop(ctx, "unknown"); err != nil {
		t.Fatal("stopping unknown job must not error")
	}
}
