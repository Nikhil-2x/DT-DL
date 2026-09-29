CREATE TABLE datasets (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    bucket            TEXT NOT NULL,
    key               TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE jobs (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    dataset_id          TEXT NOT NULL DEFAULT '',
    image               TEXT NOT NULL,
    entrypoint          TEXT NOT NULL,
    data_mode           TEXT NOT NULL,
    workers             INTEGER NOT NULL,
    epochs              INTEGER NOT NULL,
    gpus                INTEGER NOT NULL DEFAULT 0,
    cpu_cores           INTEGER NOT NULL DEFAULT 0,
    memory_mb           INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL,
    error               TEXT NOT NULL DEFAULT '',
    checkpoint_location TEXT NOT NULL,
    output_location     TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    started_at          TEXT,
    finished_at         TEXT
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_dataset ON jobs(dataset_id);
