// Package job holds the training-job model, its lifecycle state machine, and
// the service that drives jobs through a runner.Runner.
package job

import (
	"context"
	"errors"
	"time"
)

type Status string

const (
	StatusPending   Status = "PENDING" // created, not started
	StatusQueued    Status = "QUEUED"  // accepted, being handed to the runner
	StatusRunning   Status = "RUNNING"
	StatusStopping  Status = "STOPPING"
	StatusStopped   Status = "STOPPED"
	StatusCompleted Status = "COMPLETED"
	StatusFailed    Status = "FAILED"
)

// transitions is the whole lifecycle:
//
//	PENDING -> QUEUED -> RUNNING -> COMPLETED
//	                        |  \--> FAILED
//	                        \-----> STOPPING -> STOPPED
//
// QUEUED can also fail or be stopped; STOPPING can still resolve to
// COMPLETED/FAILED if the run ends before the stop lands.
var transitions = map[Status][]Status{
	StatusPending:  {StatusQueued},
	StatusQueued:   {StatusRunning, StatusFailed, StatusStopping},
	StatusRunning:  {StatusCompleted, StatusFailed, StatusStopping},
	StatusStopping: {StatusStopped, StatusCompleted, StatusFailed},
}

func CanTransition(from, to Status) bool {
	for _, t := range transitions[from] {
		if t == to {
			return true
		}
	}
	return false
}

func (s Status) Terminal() bool {
	return s == StatusStopped || s == StatusCompleted || s == StatusFailed
}

type Job struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	DatasetID  string `json:"dataset_id,omitempty"`
	Image      string `json:"image"`
	Entrypoint string `json:"entrypoint"`
	DataMode   string `json:"data_mode"`

	Workers  int `json:"workers"`
	Epochs   int `json:"epochs"`
	GPUs     int `json:"gpus"`
	CPUCores int `json:"cpu_cores"`
	MemoryMB int `json:"memory_mb"`

	Status Status `json:"status"`
	Error  string `json:"error,omitempty"`

	// train.py checkpoints to <bucket>/<run-id>/epoch_N.pt; the run id is the
	// job id. The final model is the last checkpoint.
	CheckpointLocation string `json:"checkpoint_location"`
	OutputLocation     string `json:"output_location,omitempty"`

	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
}

var ErrNotFound = errors.New("job not found")

type Repository interface {
	Create(ctx context.Context, j Job) error
	Get(ctx context.Context, id string) (Job, error) // ErrNotFound if missing
	List(ctx context.Context) ([]Job, error)         // newest first
	Update(ctx context.Context, j Job) error         // ErrNotFound if missing
	ListByStatus(ctx context.Context, statuses ...Status) ([]Job, error)
	// CountUnfinishedByDataset counts non-terminal jobs using the dataset.
	CountUnfinishedByDataset(ctx context.Context, datasetID string) (int, error)
}
