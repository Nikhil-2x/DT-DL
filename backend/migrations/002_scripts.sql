CREATE TABLE scripts (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    bucket            TEXT NOT NULL,
    key               TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

ALTER TABLE jobs ADD COLUMN script_id TEXT NOT NULL DEFAULT '';

CREATE INDEX idx_jobs_script ON jobs(script_id);
