// Package script holds user-supplied training projects: one Python file that
// follows the plug-in contract (build_model, get_dataset) documented in
// ml-runner/project_loader.py. The platform stores the file in object storage
// and the training container imports it; the control plane never executes it.
package script

import (
	"context"
	_ "embed"
	"errors"
	"time"
)

// Template is a runnable starting point served by GET /api/v1/scripts/template.
//
//go:embed template.py
var Template string

// ObjectName is the fixed object name inside a script's key prefix. The
// client-supplied filename is kept for display only and never used in a key.
const ObjectName = "project.py"

type Script struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	OriginalFilename string    `json:"original_filename"`
	Bucket           string    `json:"bucket"`
	Key              string    `json:"key"`
	SizeBytes        int64     `json:"size_bytes"`
	CreatedAt        time.Time `json:"created_at"`
}

var ErrNotFound = errors.New("script not found")

type Repository interface {
	Create(ctx context.Context, s Script) error
	Get(ctx context.Context, id string) (Script, error) // ErrNotFound if missing
	List(ctx context.Context) ([]Script, error)         // newest first
	Delete(ctx context.Context, id string) error        // ErrNotFound if missing
}

// UsageChecker reports whether a script is referenced by unfinished jobs.
// The job service satisfies it and is injected at startup.
type UsageChecker interface {
	ScriptInUse(ctx context.Context, scriptID string) (bool, error)
}
