// Package runner defines JobRunner, the seam between the control plane and
// whatever actually executes training. The job service only ever talks to
// this interface, so adding a KubernetesRunner later does not touch the API
// or service layers.
package runner

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
)

var ErrNotFound = errors.New("runner: job not found")

// AllowedEntrypoints is the closed set of scripts a job may run. Clients
// cannot supply commands; they only pick parameters of a known script.
var AllowedEntrypoints = map[string]bool{"train.py": true}

var validDataModes = map[string]bool{"synthetic": true, "cifar10": true, "sharded": true}

// safeToken matches values that are safe to pass as a single CLI argument or
// container name component: no leading dash, no whitespace or shell syntax.
var safeToken = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/:@-]*$`)

// TrainingSpec is the structured description of one training run. Runners
// translate it into an execution mechanism; nothing is ever concatenated
// into a shell string.
//
// It mirrors the arguments of ml-runner/train.py:
//
//	torchrun --nproc_per_node=<Workers> <Entrypoint> --epochs <Epochs>
//	    --run-id <RunID> --storage s3 --s3-bucket <CheckpointBucket> ...
type TrainingSpec struct {
	JobID      string
	RunID      string // train.py --run-id; also the checkpoint prefix
	Image      string
	Entrypoint string
	Workers    int // torchrun --nproc_per_node
	Epochs     int
	DataMode   string // synthetic | cifar10 | sharded

	// Only for DataMode == "sharded".
	DatasetBucket string
	DatasetPrefix string

	CheckpointBucket string
	Resources        Resources
}

type Resources struct {
	CPUCores int // 0 = unlimited
	MemoryMB int // 0 = unlimited
	GPUs     int // per worker
}

func (s TrainingSpec) Validate() error {
	for name, v := range map[string]string{"job id": s.JobID, "run id": s.RunID, "image": s.Image, "checkpoint bucket": s.CheckpointBucket} {
		if !safeToken.MatchString(v) {
			return fmt.Errorf("invalid %s %q", name, v)
		}
	}
	if !AllowedEntrypoints[s.Entrypoint] {
		return fmt.Errorf("entrypoint %q is not allowed", s.Entrypoint)
	}
	if !validDataModes[s.DataMode] {
		return fmt.Errorf("invalid data mode %q", s.DataMode)
	}
	if s.Workers < 1 || s.Epochs < 1 {
		return errors.New("workers and epochs must be >= 1")
	}
	if s.Resources.CPUCores < 0 || s.Resources.MemoryMB < 0 || s.Resources.GPUs < 0 {
		return errors.New("resources must not be negative")
	}
	if s.DataMode == "sharded" {
		if !safeToken.MatchString(s.DatasetBucket) || !safeToken.MatchString(s.DatasetPrefix) {
			return errors.New("sharded data mode needs a valid dataset bucket and prefix")
		}
	}
	return nil
}

type State string

const (
	StateRunning   State = "RUNNING"
	StateSucceeded State = "SUCCEEDED"
	StateFailed    State = "FAILED"
)

type Status struct {
	State   State
	Message string // human-readable detail for FAILED
}

type JobRunner interface {
	// Start launches the run and returns once it has been submitted (not
	// when it finishes).
	Start(ctx context.Context, spec TrainingSpec) error
	// Stop asks the run to terminate. Stopping an unknown job is not an error.
	Stop(ctx context.Context, jobID string) error
	// Status returns ErrNotFound if the runner has no record of the job.
	Status(ctx context.Context, jobID string) (Status, error)
	// Logs returns the last `tail` lines (all if tail <= 0); ErrNotFound if
	// the runner has no record of the job.
	Logs(ctx context.Context, jobID string, tail int) (io.ReadCloser, error)
}

// TODO(KubernetesRunner): implement JobRunner with client-go. Sketch:
//   - Start: create a headless Service + an Indexed Job (or PyTorchJob via
//     the Kubeflow training operator) with one pod per worker; rendezvous
//     via torchrun --rdzv-backend=c10d --rdzv-endpoint=<svc>:29400;
//     Resources map to pod requests/limits (nvidia.com/gpu for GPUs); S3
//     credentials come from a Secret, not CLI args.
//   - Status: derive from Job/pod conditions (or watch instead of poll).
//   - Stop: delete the Job with foreground propagation.
//   - Logs: pod log stream of the rank-0 pod.
// See docs/architecture.md.
