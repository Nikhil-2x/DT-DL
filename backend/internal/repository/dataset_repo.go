package repository

import (
	"context"
	"database/sql"
	"errors"

	"dtdl/backend/internal/dataset"
)

type DatasetRepo struct{ db *sql.DB }

func NewDatasetRepo(db *sql.DB) *DatasetRepo { return &DatasetRepo{db: db} }

const datasetCols = `id, name, original_filename, bucket, key, size_bytes, status, created_at, updated_at`

func (r *DatasetRepo) Create(ctx context.Context, d dataset.Dataset) error {
	_, err := r.db.ExecContext(ctx, `INSERT INTO datasets (`+datasetCols+`) VALUES (?,?,?,?,?,?,?,?,?)`,
		d.ID, d.Name, d.OriginalFilename, d.Bucket, d.Key, d.SizeBytes, string(d.Status),
		formatTime(d.CreatedAt), formatTime(d.UpdatedAt))
	return err
}

func (r *DatasetRepo) Get(ctx context.Context, id string) (dataset.Dataset, error) {
	d, err := scanDataset(r.db.QueryRowContext(ctx, `SELECT `+datasetCols+` FROM datasets WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return dataset.Dataset{}, dataset.ErrNotFound
	}
	return d, err
}

func (r *DatasetRepo) List(ctx context.Context) ([]dataset.Dataset, error) {
	rows, err := r.db.QueryContext(ctx, `SELECT `+datasetCols+` FROM datasets ORDER BY created_at DESC, id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []dataset.Dataset{}
	for rows.Next() {
		d, err := scanDataset(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (r *DatasetRepo) Update(ctx context.Context, d dataset.Dataset) error {
	res, err := r.db.ExecContext(ctx,
		`UPDATE datasets SET name=?, original_filename=?, bucket=?, key=?, size_bytes=?, status=?, updated_at=? WHERE id=?`,
		d.Name, d.OriginalFilename, d.Bucket, d.Key, d.SizeBytes, string(d.Status), formatTime(d.UpdatedAt), d.ID)
	return expectOne(res, err, dataset.ErrNotFound)
}

func (r *DatasetRepo) Delete(ctx context.Context, id string) error {
	res, err := r.db.ExecContext(ctx, `DELETE FROM datasets WHERE id = ?`, id)
	return expectOne(res, err, dataset.ErrNotFound)
}

type scanner interface{ Scan(dest ...any) error }

func scanDataset(s scanner) (dataset.Dataset, error) {
	var d dataset.Dataset
	var status, created, updated string
	if err := s.Scan(&d.ID, &d.Name, &d.OriginalFilename, &d.Bucket, &d.Key, &d.SizeBytes, &status, &created, &updated); err != nil {
		return dataset.Dataset{}, err
	}
	var err error
	d.Status = dataset.Status(status)
	if d.CreatedAt, err = parseTime(created); err != nil {
		return dataset.Dataset{}, err
	}
	if d.UpdatedAt, err = parseTime(updated); err != nil {
		return dataset.Dataset{}, err
	}
	return d, nil
}

func expectOne(res sql.Result, err error, notFound error) error {
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return notFound
	}
	return nil
}
