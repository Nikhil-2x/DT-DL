package repository

import (
	"context"
	"database/sql"
	"errors"
	"strings"

	"dtdl/backend/internal/job"
)

type JobRepo struct{ db *sql.DB }

func NewJobRepo(db *sql.DB) *JobRepo { return &JobRepo{db: db} }

const jobCols = `id, name, dataset_id, image, entrypoint, data_mode, workers, epochs, gpus, cpu_cores, memory_mb,
	status, error, checkpoint_location, output_location, created_at, updated_at, started_at, finished_at`

func (r *JobRepo) Create(ctx context.Context, j job.Job) error {
	_, err := r.db.ExecContext(ctx, `INSERT INTO jobs (`+jobCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		j.ID, j.Name, j.DatasetID, j.Image, j.Entrypoint, j.DataMode, j.Workers, j.Epochs, j.GPUs, j.CPUCores, j.MemoryMB,
		string(j.Status), j.Error, j.CheckpointLocation, j.OutputLocation,
		formatTime(j.CreatedAt), formatTime(j.UpdatedAt), nullTime(j.StartedAt), nullTime(j.FinishedAt))
	return err
}

func (r *JobRepo) Get(ctx context.Context, id string) (job.Job, error) {
	j, err := scanJob(r.db.QueryRowContext(ctx, `SELECT `+jobCols+` FROM jobs WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return job.Job{}, job.ErrNotFound
	}
	return j, err
}

func (r *JobRepo) List(ctx context.Context) ([]job.Job, error) {
	return r.query(ctx, `SELECT `+jobCols+` FROM jobs ORDER BY created_at DESC, id`)
}

func (r *JobRepo) ListByStatus(ctx context.Context, statuses ...job.Status) ([]job.Job, error) {
	if len(statuses) == 0 {
		return []job.Job{}, nil
	}
	args := make([]any, len(statuses))
	for i, s := range statuses {
		args[i] = string(s)
	}
	q := `SELECT ` + jobCols + ` FROM jobs WHERE status IN (?` + strings.Repeat(",?", len(statuses)-1) + `) ORDER BY created_at, id`
	return r.query(ctx, q, args...)
}

func (r *JobRepo) Update(ctx context.Context, j job.Job) error {
	res, err := r.db.ExecContext(ctx, `UPDATE jobs SET status=?, error=?, output_location=?, updated_at=?, started_at=?, finished_at=? WHERE id=?`,
		string(j.Status), j.Error, j.OutputLocation, formatTime(j.UpdatedAt), nullTime(j.StartedAt), nullTime(j.FinishedAt), j.ID)
	return expectOne(res, err, job.ErrNotFound)
}

func (r *JobRepo) CountUnfinishedByDataset(ctx context.Context, datasetID string) (int, error) {
	var n int
	err := r.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs WHERE dataset_id = ? AND status NOT IN ('STOPPED','COMPLETED','FAILED')`, datasetID).Scan(&n)
	return n, err
}

func (r *JobRepo) query(ctx context.Context, q string, args ...any) ([]job.Job, error) {
	rows, err := r.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []job.Job{}
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

func scanJob(s scanner) (job.Job, error) {
	var j job.Job
	var status, created, updated string
	var started, finished sql.NullString
	if err := s.Scan(&j.ID, &j.Name, &j.DatasetID, &j.Image, &j.Entrypoint, &j.DataMode, &j.Workers, &j.Epochs,
		&j.GPUs, &j.CPUCores, &j.MemoryMB, &status, &j.Error, &j.CheckpointLocation, &j.OutputLocation,
		&created, &updated, &started, &finished); err != nil {
		return job.Job{}, err
	}
	var err error
	j.Status = job.Status(status)
	if j.CreatedAt, err = parseTime(created); err != nil {
		return job.Job{}, err
	}
	if j.UpdatedAt, err = parseTime(updated); err != nil {
		return job.Job{}, err
	}
	if j.StartedAt, err = scanNullTime(started); err != nil {
		return job.Job{}, err
	}
	if j.FinishedAt, err = scanNullTime(finished); err != nil {
		return job.Job{}, err
	}
	return j, nil
}
