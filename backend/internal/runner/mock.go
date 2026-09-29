package runner

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// Mock simulates training without running anything: a job "runs" for
// Epochs*EpochDuration and then succeeds. It emits log lines shaped like
// train.py's output. Used for local demos and tests.
type Mock struct {
	EpochDuration time.Duration
	Now           func() time.Time // injectable clock for tests

	mu   sync.Mutex
	jobs map[string]*mockJob
}

type mockJob struct {
	spec    TrainingSpec
	started time.Time
	stopped time.Time // zero unless stopped
}

func NewMock(epochDuration time.Duration) *Mock {
	return &Mock{EpochDuration: epochDuration, Now: time.Now, jobs: map[string]*mockJob{}}
}

func (m *Mock) Start(_ context.Context, spec TrainingSpec) error {
	if err := spec.Validate(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.jobs[spec.JobID]; exists {
		return fmt.Errorf("job %s already started", spec.JobID)
	}
	m.jobs[spec.JobID] = &mockJob{spec: spec, started: m.Now()}
	return nil
}

func (m *Mock) Stop(_ context.Context, jobID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if j, ok := m.jobs[jobID]; ok && j.stopped.IsZero() {
		j.stopped = m.Now()
	}
	return nil
}

func (m *Mock) Status(_ context.Context, jobID string) (Status, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[jobID]
	if !ok {
		return Status{}, ErrNotFound
	}
	return m.statusLocked(j), nil
}

func (m *Mock) statusLocked(j *mockJob) Status {
	if !j.stopped.IsZero() {
		return Status{State: StateFailed, Message: "stopped"}
	}
	if m.Now().Sub(j.started) >= time.Duration(j.spec.Epochs)*m.EpochDuration {
		return Status{State: StateSucceeded}
	}
	return Status{State: StateRunning}
}

func (m *Mock) Logs(_ context.Context, jobID string, tail int) (io.ReadCloser, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[jobID]
	if !ok {
		return nil, ErrNotFound
	}
	end := m.Now()
	if !j.stopped.IsZero() {
		end = j.stopped
	}
	done := 0
	if m.EpochDuration > 0 {
		done = int(end.Sub(j.started) / m.EpochDuration)
	}
	done = min(done, j.spec.Epochs)

	lines := []string{fmt.Sprintf("[mock] starting %d worker(s), %d epoch(s), run-id=%s", j.spec.Workers, j.spec.Epochs, j.spec.RunID)}
	for e := 0; e < done; e++ {
		for r := 0; r < j.spec.Workers; r++ {
			lines = append(lines, fmt.Sprintf("[rank %d] epoch %d avg_loss=%.4f param_checksum=%.4f time=%.2fs",
				r, e, 2.3/float64(e+1)+float64(r)*0.01, 12.3456+float64(e)*0.1, m.EpochDuration.Seconds()))
		}
	}
	if !j.stopped.IsZero() {
		lines = append(lines, "[mock] stopped by user")
	}
	if tail > 0 && len(lines) > tail {
		lines = lines[len(lines)-tail:]
	}
	return io.NopCloser(strings.NewReader(strings.Join(lines, "\n") + "\n")), nil
}
