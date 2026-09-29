package runner

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
)

// ExecFunc runs a command and returns its combined output. Injectable so
// tests need no Docker daemon.
type ExecFunc func(ctx context.Context, name string, args ...string) ([]byte, error)

type DockerConfig struct {
	Binary string // default "docker"
	// S3 endpoint/credentials handed to train.py. The endpoint must be
	// reachable from inside the container (e.g. host.docker.internal).
	S3Endpoint  string
	S3AccessKey string
	S3SecretKey string
}

// Docker runs each job as a detached container on the local Docker daemon,
// exactly like the README's manual command:
//
//	docker run ml-runner:dev torchrun --nproc_per_node=4 train.py --epochs 3 --run-id X
//
// It uses exec.Command with an argv slice, never a shell.
type Docker struct {
	cfg  DockerConfig
	exec ExecFunc
}

func NewDocker(cfg DockerConfig) *Docker { return NewDockerWithExec(cfg, defaultExec) }

func NewDockerWithExec(cfg DockerConfig, exec ExecFunc) *Docker {
	if cfg.Binary == "" {
		cfg.Binary = "docker"
	}
	return &Docker{cfg: cfg, exec: exec}
}

func containerName(jobID string) string { return "dtdl-job-" + jobID }

func (d *Docker) Start(ctx context.Context, spec TrainingSpec) error {
	if err := spec.Validate(); err != nil {
		return err
	}
	if spec.Resources.GPUs > 0 {
		return fmt.Errorf("gpus are not supported by the docker runner (CPU/gloo image)")
	}
	if _, err := d.exec(ctx, d.cfg.Binary, d.RunArgs(spec)...); err != nil {
		return err
	}
	return nil
}

// RunArgs builds the `docker run` argv for a spec. Exported for tests.
func (d *Docker) RunArgs(spec TrainingSpec) []string {
	args := []string{
		"run", "-d",
		"--name", containerName(spec.JobID),
		"--label", "dtdl.job-id=" + spec.JobID,
		"--add-host", "host.docker.internal:host-gateway",
	}
	if spec.Resources.CPUCores > 0 {
		args = append(args, "--cpus", strconv.Itoa(spec.Resources.CPUCores))
	}
	if spec.Resources.MemoryMB > 0 {
		args = append(args, "--memory", strconv.Itoa(spec.Resources.MemoryMB)+"m")
	}
	args = append(args,
		spec.Image,
		"torchrun", "--nproc_per_node="+strconv.Itoa(spec.Workers),
		spec.Entrypoint,
		"--epochs", strconv.Itoa(spec.Epochs),
		"--run-id", spec.RunID,
		"--storage", "s3",
		"--s3-endpoint", d.cfg.S3Endpoint,
		"--s3-bucket", spec.CheckpointBucket,
		"--s3-access-key", d.cfg.S3AccessKey,
		"--s3-secret-key", d.cfg.S3SecretKey,
		"--data", spec.DataMode,
	)
	if spec.DataMode == "sharded" {
		args = append(args,
			"--dataset-bucket", spec.DatasetBucket,
			"--dataset-prefix", spec.DatasetPrefix,
			"--dataset-local-dir", "/tmp/shard_data",
		)
	}
	return args
}

func (d *Docker) Stop(ctx context.Context, jobID string) error {
	_, err := d.exec(ctx, d.cfg.Binary, "stop", "-t", "30", containerName(jobID))
	if isNoSuchContainer(err) {
		return nil
	}
	return err
}

func (d *Docker) Status(ctx context.Context, jobID string) (Status, error) {
	out, err := d.exec(ctx, d.cfg.Binary, "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", containerName(jobID))
	if isNoSuchContainer(err) {
		return Status{}, ErrNotFound
	}
	if err != nil {
		return Status{}, err
	}
	fields := strings.Fields(string(out))
	if len(fields) != 2 {
		return Status{}, fmt.Errorf("unexpected docker inspect output %q", strings.TrimSpace(string(out)))
	}
	code, _ := strconv.Atoi(fields[1])
	switch fields[0] {
	case "created", "running", "restarting", "paused", "removing":
		return Status{State: StateRunning}, nil
	case "exited":
		if code == 0 {
			return Status{State: StateSucceeded}, nil
		}
		return Status{State: StateFailed, Message: fmt.Sprintf("training exited with code %d", code)}, nil
	default: // dead, unknown
		return Status{State: StateFailed, Message: "container state: " + fields[0]}, nil
	}
}

func (d *Docker) Logs(ctx context.Context, jobID string, tail int) (io.ReadCloser, error) {
	args := []string{"logs"}
	if tail > 0 {
		args = append(args, "--tail", strconv.Itoa(tail))
	}
	args = append(args, containerName(jobID))
	out, err := d.exec(ctx, d.cfg.Binary, args...)
	if isNoSuchContainer(err) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return io.NopCloser(bytes.NewReader(out)), nil
}

func isNoSuchContainer(err error) bool {
	return err != nil && strings.Contains(err.Error(), "No such")
}

// defaultExec includes the subcommand and stderr in errors but never the
// full argument list, which contains S3 credentials.
func defaultExec(ctx context.Context, name string, args ...string) ([]byte, error) {
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	if err != nil {
		sub := ""
		if len(args) > 0 {
			sub = args[0]
		}
		return out, fmt.Errorf("%s %s: %w: %s", name, sub, err, strings.TrimSpace(string(out)))
	}
	return out, nil
}
