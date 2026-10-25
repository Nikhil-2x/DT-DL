package repository

import (
	"context"
	"database/sql"
	"errors"

	"dtdl/backend/internal/script"
)

type ScriptRepo struct{ db *sql.DB }

func NewScriptRepo(db *sql.DB) *ScriptRepo { return &ScriptRepo{db: db} }

const scriptCols = `id, name, original_filename, bucket, key, size_bytes, created_at`

func (r *ScriptRepo) Create(ctx context.Context, s script.Script) error {
	_, err := r.db.ExecContext(ctx, `INSERT INTO scripts (`+scriptCols+`) VALUES (?,?,?,?,?,?,?)`,
		s.ID, s.Name, s.OriginalFilename, s.Bucket, s.Key, s.SizeBytes, formatTime(s.CreatedAt))
	return err
}

func (r *ScriptRepo) Get(ctx context.Context, id string) (script.Script, error) {
	s, err := scanScript(r.db.QueryRowContext(ctx, `SELECT `+scriptCols+` FROM scripts WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return script.Script{}, script.ErrNotFound
	}
	return s, err
}

func (r *ScriptRepo) List(ctx context.Context) ([]script.Script, error) {
	rows, err := r.db.QueryContext(ctx, `SELECT `+scriptCols+` FROM scripts ORDER BY created_at DESC, id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []script.Script{}
	for rows.Next() {
		s, err := scanScript(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func (r *ScriptRepo) Delete(ctx context.Context, id string) error {
	res, err := r.db.ExecContext(ctx, `DELETE FROM scripts WHERE id = ?`, id)
	return expectOne(res, err, script.ErrNotFound)
}

func scanScript(s scanner) (script.Script, error) {
	var sc script.Script
	var created string
	if err := s.Scan(&sc.ID, &sc.Name, &sc.OriginalFilename, &sc.Bucket, &sc.Key, &sc.SizeBytes, &created); err != nil {
		return script.Script{}, err
	}
	var err error
	if sc.CreatedAt, err = parseTime(created); err != nil {
		return script.Script{}, err
	}
	return sc, nil
}
